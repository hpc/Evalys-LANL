# coding: utf-8

import functools

import matplotlib.dates
import matplotlib.patches
import numpy
import pandas
import os

from . import core
from .. import utils


def NOLABEL(_):
    """Labeler strategy disabling the labeling of jobs."""
    return ""


class GanttVisualization(core.Visualization):
    """
    Visualization of a jobset as a Gantt chart.

    The `GanttVisualization` class displays a jobset as a Gantt chart.  Each
    job in the jobset is represented as a set of rectangle.
    The x-axis represents time, while the y-axis represents resources.

    :cvar COLUMNS: The columns required to build the visualization.

    :ivar _lspec: The specification of the layout for the visualization.
    :vartype _lspec: `core._LayoutSpec`

    :ivar _ax: The `Axe` to draw on.

    :ivar palette: The palette of colors to be used.

    :ivar xscale:
        The requested adaptation of the x-axis scale.
        Valid values are `None`, and `'time'`.

        * It defaults to `None`, and uses raw values by default.
        * If set to `time`, the x-axis interprets the data as timestamps, and
          uses a time-aware semantic.

    :ivar alpha:
        The transparency level of the rectangles depicting jobs.  It defaults
        to `0.4`.
    :vartype alpha: float

    :ivar colorer:
        The strategy to assign a color to a job.  By default, the colors of the
        palette are picked with a round-robin strategy.
        The colorer is a function expecting two positional parameters: first
        the `job`, and second the `palette`.
        See `GanttVisualization.round_robin_map` for an example.

    :ivar labeler:
        The strategy to label jobs.  By default, the `jobID` column is used to
        label jobs.
        To disable the labeling of jobs, use :func:`~gantt.NOLABEL`.
    """

    COLUMNS = (
        "jobID",
        "allocated_resources",
        "execution_time",
        "finish_time",
        "starting_time",
        "submission_time",
        "purpose",
    )

    def __init__(self, lspec, *, title="Gantt chart"):
        super().__init__(lspec)
        self.title = title
        self.xscale = None
        self.alpha = 0.4
        self.colorer = self.round_robin_map
        self.labeler = lambda job: str(job["jobID"])
        self._columns = self.COLUMNS

    def _customize_layout(self):
        self._ax.grid(True)

        # adapt scale of axes if requested
        if self.xscale == "time":
            self._ax.xaxis.set_major_formatter(
                matplotlib.dates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
            )

    @staticmethod
    def _adapt_uniq_num(df):
        """
        Assigns each job in the df a unique number - effectively an internal jobid
        :param df:
        :return:
        """
        df["uniq_num"] = numpy.arange(0, len(df))

    @staticmethod
    def _adapt_time_xscale(df):
        # interpret columns with time aware semantics
        df["submission_time"] = pandas.to_datetime(df["submission_time"], unit="s")
        df["starting_time"] = pandas.to_datetime(df["starting_time"], unit="s")
        df["execution_time"] = pandas.to_timedelta(df["execution_time"], unit="s")
        df["finish_time"] = df["starting_time"] + df["execution_time"]
        # convert columns to use them with matplotlib
        df["submission_time"] = df["submission_time"].map(matplotlib.dates.date2num)
        df["starting_time"] = df["starting_time"].map(matplotlib.dates.date2num)
        df["finish_time"] = df["finish_time"].map(matplotlib.dates.date2num)
        df["execution_time"] = df["finish_time"] - df["starting_time"]

    def _adapt(self, df):
        self._adapt_uniq_num(df)
        if self.xscale == "time":
            self._adapt_time_xscale(df)

    @staticmethod
    def _annotate(rect, label):
        rx, ry = rect.get_xy()
        cx, cy = rx + rect.get_width() / 2.0, ry + rect.get_height() / 2.0
        rect.axes.annotate(
            label, (cx, cy), color="black", fontsize=1, ha="center", va="center"
        )

    @staticmethod
    def round_robin_map(job, palette):
        return palette[job["uniq_num"] % len(palette)]

    @staticmethod
    def project_color_map(job, palette):
        return palette[job["account"]-1]

    @staticmethod
    def partition_color_map(job, palette):
        return palette[job["partition"]]

    @staticmethod
    def dependency_color_map(job, palette):
        return palette[job["dependency_chain_head"]  % len(palette)]

    @staticmethod
    def user_color_map(job, palette):
        return palette[job["user"]]

    @staticmethod
    def top_user_color_map(job, palette):
        return palette[job["user_id"]]

    def _coloration_middleman(self, job, x0, duration, height, itv, colorationMethod="default", num_projects=None, num_top_users=None, partition_count=0, num_users=None):
        coloration_methods = {
            "default": self._return_default_rectangle,
            "project": self._return_project_rectangle if num_projects is not None else None,
            "dependency": self._return_dependency_rectangle,
            "user": self._return_user_rectangle if num_users is not None else None,
            "user_top_20" : self._return_top_user_rectangle if num_top_users is not None else None,
            "sched" : self._return_sched_rectangle,
            "wait" : self._return_wait_rectangle,
            "partition" : self._return_partition_rectangle,

        }

        # TODO Does this even work?
        method_func = coloration_methods.get(colorationMethod, self._return_default_rectangle)
        return method_func(job, x0, duration, height, itv, num_projects, num_users, num_top_users, partition_count)

    def _return_default_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                  num_top_users=None, partition_count=None):
        return self._create_rectangle(job, x0, duration, height, itv, self.colorer)

    def _return_project_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                  num_top_users=None, partition_count=None):
        edge_color = "#FF0400" if "SchedBackfill" in job["flags"] else "#00E1FF" if "SchedSubmit" in job[
            "flags"] else "black"
        return self._create_rectangle(job, x0, duration, height, itv, self.project_color_map, edge_color=edge_color,
                                      palette=core.generate_palette(num_projects))

    def _return_dependency_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                     num_top_users=None, partition_count=None):
        return self._create_rectangle(job, x0, duration, height, itv, self.dependency_color_map,
                                      palette=core.generate_palette(8))

    def _return_user_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                               num_top_users=None, partition_count=None):
        return self._create_rectangle(job, x0, duration, height, itv, self.user_color_map,
                                      palette=core.generate_palette(num_users + 1))

    def _return_top_user_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                   num_top_users=None, partition_count=None):
        if job["user_id"] != 0:
            return self._create_rectangle(job, x0, duration, height, itv, self.top_user_color_map,
                                          palette=core.generate_palette(num_top_users))
        else:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#C2C2C2", facecolor="#C2C2C2")

    def _return_sched_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                num_top_users=None, partition_count=None):
        if "SchedBackfill" in job["flags"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#7A0200", facecolor="#7A0200")
        elif "SchedSubmit" in job["flags"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#246A73", facecolor="#246A73")
        else:
            return self._create_rectangle(job, x0, duration, height, itv, self.colorer)

    def _return_wait_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                               num_top_users=None, partition_count=None):
        return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#95374F",
                                      alpha=job["normalized_eligible_wait"], facecolor="#95374F")

    def _return_partition_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                    num_top_users=None, partition_count=None):
        if "SchedBackfill" in job["flags"]:
            edge_color = "#FF0400"
        elif "SchedSubmit" in job["flags"]:
            edge_color = "#00E1FF"
        else:
            edge_color = "black"
        return self._create_rectangle(job, x0, duration, height, itv, self.partition_color_map,
                                      alpha=job["normalized_account"], edge_color=edge_color, palette=core.generate_palette(partition_count))

    def _create_rectangle(self, job, x0, duration, height, itv, color_func, alpha=-1, edge_color="black", palette=None, facecolor=None):
        if alpha == -1:
            alpha = self.alpha
        if palette == None:
            palette = self.palette
        if facecolor is None:
            return matplotlib.patches.Rectangle(
                (x0, itv.inf),
                duration,
                height,
                alpha=alpha,
                facecolor=functools.partial(color_func, palette=palette)(job),
                edgecolor=edge_color,
                linewidth=0.5,
            )
        else:
            return matplotlib.patches.Rectangle(
                (x0, itv.inf),
                duration,
                height,
                alpha=alpha,
                facecolor=facecolor,
                edgecolor=edge_color,
                linewidth=0.5,
            )


    def _draw(
        self, df, resvStart=None, resvExecTime=None, resvNodes=None, resvSet=None, colorationMethod="default", num_projects=None, num_users=None, num_top_users=None, partition_count=0,
    ):
        def _plot_job(job, colorationMethod="default", num_projects=None, num_top_users=None, partition_count=0):
            x0 = job["starting_time"]
            duration = job["execution_time"]
            if job["purpose"] != "reservation":
                for itv in job["allocated_resources"].intervals():
                    height = itv.sup - itv.inf + 1
                    rect = self._coloration_middleman(job, x0, duration, height, itv, colorationMethod, num_projects, num_top_users, partition_count, num_users)
                    # TODO This is disgusting. Please use functions.
                    # if colorationMethod == "default":
                    #     rect = matplotlib.patches.Rectangle(
                    #         (x0, itv.inf),
                    #         duration,
                    #         height,
                    #         alpha=self.alpha,
                    #
                    #         facecolor=functools.partial(self.colorer, palette=self.palette)(
                    #             job
                    #         ),
                    #         edgecolor="black",
                    #         linewidth=0.5,
                    #     )
                    # elif colorationMethod == "project" and num_projects != None:
                    #     if "SchedBackfill" in job["flags"]:
                    #         edge_color= "#FF0400"
                    #     elif "SchedSubmit" in job["flags"]:
                    #         edge_color= "#00E1FF"
                    #     else:
                    #         edge_color= "black"
                    #     rect = matplotlib.patches.Rectangle(
                    #         (x0, itv.inf),
                    #         duration,
                    #         height,
                    #         alpha=self.alpha,
                    #
                    #         facecolor=functools.partial(self.project_color_map, palette=core.generate_palette(num_projects))(
                    #             job
                    #         ),
                    #         edgecolor=edge_color,
                    #         linewidth=0.5,
                    #     )
                    # elif colorationMethod == "dependency":
                    #     rect = matplotlib.patches.Rectangle(
                    #         (x0, itv.inf),
                    #         duration,
                    #         height,
                    #         alpha=self.alpha,
                    #
                    #         facecolor=functools.partial(self.dependency_color_map, palette=core.generate_palette(8))(
                    #             job
                    #         ),
                    #         edgecolor="black",
                    #         linewidth=0.5,
                    #     )
                    # elif colorationMethod == "user" and num_users != None:
                    #     rect = matplotlib.patches.Rectangle(
                    #         (x0, itv.inf),
                    #         duration,
                    #         height,
                    #         alpha=self.alpha,
                    #
                    #         facecolor=functools.partial(self.user_color_map, palette=core.generate_palette(num_users+1))(
                    #             job
                    #         ),
                    #         edgecolor="black",
                    #         linewidth=0.5,
                    #     )
                    # elif colorationMethod == "user_top_20" and num_top_users != None:
                    #     if job["user_id"] != 0:
                    #         rect = matplotlib.patches.Rectangle(
                    #             (x0, itv.inf),
                    #             duration,
                    #             height,
                    #             alpha=self.alpha,
                    #             facecolor=functools.partial(self.top_user_color_map, palette=core.generate_palette(num_top_users))(
                    #                 job
                    #             ),
                    #             edgecolor="black",
                    #             linewidth=0.5,
                    #         )
                    #     else:
                    #         rect = matplotlib.patches.Rectangle(
                    #             (x0, itv.inf),
                    #             duration,
                    #             height,
                    #             alpha=self.alpha,
                    #             facecolor="#C2C2C2",
                    #             edgecolor="black",
                    #             linewidth=0.5,
                    #         )
                    # elif colorationMethod == "sched":
                    #     if "SchedBackfill" in job["flags"]:
                    #         rect = matplotlib.patches.Rectangle(
                    #             (x0, itv.inf),
                    #             duration,
                    #             height,
                    #             alpha=0.8,
                    #             facecolor="#7A0200",
                    #             edgecolor="black",
                    #             linewidth=0.5,
                    #         )
                    #     elif "SchedSubmit" in job["flags"]:
                    #         rect = matplotlib.patches.Rectangle(
                    #             (x0, itv.inf),
                    #             duration,
                    #             height,
                    #             alpha=0.8,
                    #             facecolor="#246A73",
                    #             edgecolor="black",
                    #             linewidth=0.5,
                    #         )
                        # elif "SchedMain" in job["flags"]:
                        #     rect = matplotlib.patches.Rectangle(
                        #         (x0, itv.inf),
                        #         duration,
                        #         height,
                        #         alpha=0.8,
                        #         facecolor="#9EDA2F",
                        #         edgecolor="black",
                        #         linewidth=0.5,
                        #     )
                        # else:
                        #     rect = matplotlib.patches.Rectangle(
                        #         (x0, itv.inf),
                        #         duration,
                        #         height,
                        #         alpha=self.alpha,
                        #         facecolor=functools.partial(self.colorer, palette=self.palette)(
                        #             job
                        #         ),
                        #         edgecolor="black",
                        #         linewidth=0.5,
                        #     )
                    # elif colorationMethod == "wait":
                    #     rect = matplotlib.patches.Rectangle(
                    #         (x0, itv.inf),
                    #         duration,
                    #         height,
                    #         alpha=job["normalized_eligible_wait"],
                    #         facecolor="#95374F",
                    #         edgecolor="black",
                    #         linewidth=0.5,
                    #     )
                    # elif colorationMethod == "partition":
                    #     if "SchedBackfill" in job["flags"]:
                    #         edge_color= "#FF0400"
                    #     elif "SchedSubmit" in job["flags"]:
                    #         edge_color= "#00E1FF"
                    #     else:
                    #         edge_color= "black"
                    #     rect = matplotlib.patches.Rectangle(
                    #         (x0, itv.inf),
                    #         duration,
                    #         height,
                    #         alpha=job["normalized_account"],
                    #
                    #         facecolor=functools.partial(self.partition_color_map,
                    #                                     palette=core.generate_palette(partition_count))(
                    #             job
                    #         ),
                    #         edgecolor=edge_color,
                    #         linewidth=0.5,
                    #     )

                    # else:
                    #     rect = matplotlib.patches.Rectangle(
                    #         (x0, itv.inf),
                    #         duration,
                    #         height,
                    #         alpha=self.alpha,
                    #
                    #         facecolor=functools.partial(self.colorer, palette=self.palette)(
                    #             job
                    #         ),
                    #         edgecolor="black",
                    #         linewidth=0.5,
                    #     )
                    self._ax.add_artist(rect)
                    if colorationMethod == "user" or colorationMethod == "user_top_20":
                        self._annotate(rect, str(job["username"]))
                    if colorationMethod == "dependency":
                        if job["dependency_chain_head"] != job["jobID"]:
                            self._annotate(rect, str(job["dependency_chain_head"]))
                    if colorationMethod == "project" or colorationMethod == "partition":
                        self._annotate(rect, job["account_name"])

                    # self._annotate(rect, self.labeler(job))

        df.apply(_plot_job, axis="columns", colorationMethod=colorationMethod, num_projects=num_projects, partition_count=partition_count, num_top_users=num_top_users)

        # If there's a single reservation:
        if (resvStart != None and resvExecTime != None) and resvSet == None:
            resvNodes = str(resvNodes).split("-")
            startNode = int(resvNodes[0])
            height = int(resvNodes[1]) - int(resvNodes[0])
            rect = matplotlib.patches.Rectangle(
                (resvStart, startNode),
                resvExecTime,
                height,
                alpha=self.alpha,
                facecolor="#FF0000",
                edgecolor="black",
                linewidth=0.5,
            )
            self._ax.add_artist(rect)
            # TODO Annotate reservation with name/type/purpose

        # If there are multiple reservations:
        elif resvSet != None:
            for row in resvSet:
                resvNodes = str(row["allocated_resources"])
                resvNodes = resvNodes.split(" ")
                for resvBlock in resvNodes:
                    pass
                    resvNodes = resvBlock.split("-")
                    startNode = int(resvNodes[0])
                    if len(resvNodes) < 2:
                        rect = matplotlib.patches.Rectangle(
                            (row["starting_time"], startNode),
                            row["execution_time"],
                            1,
                            alpha=self.alpha,
                            facecolor="#FF0000",
                            edgecolor="black",
                            linewidth=0.5,
                        )
                    else:
                        height = int(resvNodes[1]) - int(resvNodes[0])
                        rect = matplotlib.patches.Rectangle(
                            (row["starting_time"], startNode),
                            row["execution_time"],
                            height,
                            alpha=self.alpha,
                            facecolor="#FF0000",
                            edgecolor="black",
                            linewidth=0.5,
                        )
                        self._ax.add_artist(rect)
                    # TODO Annotate reservation with name/type/purpose

    def build(self, jobset):
        df = jobset.df.loc[:, self._columns]  # copy just what is needed
        self._adapt(df)  # extract the data required for the visualization
        self._customize_layout()  # prepare the layout for displaying the data
        self._draw(df)  # do the painting job

        # tweak boundaries to match the studied jobset
        self._ax.set(
            xlim=(df.submission_time.min(), df.finish_time.max()),
            ylim=(jobset.res_bounds.inf - 1, jobset.res_bounds.sup + 2),
        )

    def buildDf(
        self,
        df,
        res_bounds,
        windowStartTime,
        windowFinishTime,
        resvStart=None,
        resvExecTime=None,
        resvNodes=None,
        resvSet=None,
        colorationMethod="default",
        num_projects=None,
        num_users=None,
        num_top_users=None,
        partition_count=0,
    ):
        if colorationMethod == "project":
            df = df.loc[:, self.COLUMNS + ("account","account_name","flags",)]  # copy just what is needed
        elif colorationMethod == "dependency":
            df = df.loc[:, self.COLUMNS + ("dependency_chain_head",)]
        elif colorationMethod == "user" or colorationMethod == "user_top_20":
            df = df.loc[:, self.COLUMNS + ("user", "username","user_id",)]
        elif colorationMethod == "sched":
            df = df.loc[:, self.COLUMNS + ("flags",)]
        elif colorationMethod == "wait":
            df = df.loc[:, self.COLUMNS + ("normalized_eligible_wait",)]
        elif colorationMethod == "partition":
            df = df.loc[:, self.COLUMNS + ("partition","account","normalized_account","account_name","flags",)]
        elif colorationMethod == "exitstate":
            df = df.loc[:, self.COLUMNS + ("success",)]
        else:
            df = df.loc[:, self._columns]  # copy just what is needed
        self._adapt(df)  # extract the data required for the visualization
        self._customize_layout()  # prepare the layout for displaying the data
        self._draw(
            df, resvStart, resvExecTime, resvNodes, resvSet, colorationMethod, num_projects, num_users,num_top_users, partition_count
        )  # do the painting job
        # My axis setting method
        self._ax.set(
            xlim=(windowStartTime, windowFinishTime),
            ylim=(res_bounds.inf - 1, res_bounds.sup + 2),
        )


class DiffGanttVisualization(GanttVisualization):
    def __init__(self, lspec, *, title="Gantt charts comparison"):
        super().__init__(lspec, title=title)
        self.alpha = 0.5
        self.colorer = lambda _, palette: palette[0]  # single color per jobset
        self.labeler = NOLABEL  # do not label jobs
        self.palette = None  # let .build(…) figure the number of colors

    def build(self, jobsets):
        _orig_palette = self.palette  # save original palette

        # create an adapted palette if none has been provided
        palette = self.palette or core.generate_palette(len(jobsets))

        gxmin, gxmax = None, None  # global xlim
        captions = []  # list of proxy objects for the legend

        for idx, (js_name, js_obj) in enumerate(jobsets.items()):
            # create a palette made of a single color for current jobset
            color = palette[idx]
            self.palette = [color]

            # build as a GanttVisualization for current jobset
            super().build(js_obj)

            # tweak visualization appearance
            if idx:
                # recompute xlim with respect to previous GanttVisualization
                xmin, xmax = self._ax.get_xlim()
                gxmin, gxmax = min(xmin, gxmin), max(xmax, gxmax)
                self._ax.set_xlim(gxmin, gxmax)
            else:
                # first GanttVisualization, save xlim as is
                gxmin, gxmax = self._ax.get_xlim()

            # create a proxy object for the legend
            captions.append(
                matplotlib.patches.Patch(color=color, alpha=self.alpha, label=js_name)
            )

        # add legend to the visualization
        self._ax.legend(handles=captions, loc="best")

        self.palette = _orig_palette  # restore original palette


def plot_gantt(jobset, *, title="Gantt chart", **kwargs):
    """
    Helper function to create a Gantt chart of a workload.

    :param jobset: The jobset under study.
    :type jobset: ``JobSet``

    :param title: The title of the window.
    :type title: ``str``

    :param \**kwargs:
        The keyword arguments to be fed to the constructor of the visualization
        class.
    """
    layout = core.SimpleLayout(wtitle=title)
    plot = layout.inject(GanttVisualization, spskey="all", title=title)
    utils.bulksetattr(plot, **kwargs)
    plot.build(jobset)
    layout.show()


def plot_gantt_df(
    df,
    res_bounds,
    windowStartTime,
    windowFinishTime,
    *,
    title="Gantt chart",
    resvStart=None,
    resvExecTime=None,
    resvNodes=None,
    resvSet=None,
    dimensions=(6.4,4.8),
    colorationMethod="default",
    num_projects=None,
    num_users=None,
    num_top_users=None,
    partition_count=0,
    **kwargs
):
    """
    Helper function to create a Gantt chart of a workload.

    :param jobset: The jobset under study.
    :type jobset: ``JobSet``

    :param title: The title of the window.
    :type title: ``str``

    :param \**kwargs:
        The keyword arguments to be fed to the constructor of the visualization
        class.
    """
    layout = core.SimpleLayout(wtitle=title,dimensions=dimensions)
    plot = layout.inject(GanttVisualization, spskey="all", title=title)
    utils.bulksetattr(plot, **kwargs)
    plot.buildDf(
        df,
        res_bounds,
        windowStartTime,
        windowFinishTime,
        resvStart,
        resvExecTime,
        resvNodes,
        resvSet,
        colorationMethod,
        num_projects,
        num_users,
        num_top_users,
        partition_count,
    )
    layout.show()


def plot_diff_gantt(jobsets, *, title="Gantt charts comparison", **kwargs):
    """
    Helper function to create a comparison of Gantt charts of two (or more)
    workloads.

    :param jobsets: The jobsets under study.
    :type jobset: list(JobSet)

    :param title: The title of the window.
    :type title: str

    :param \**kwargs:
        The keyword arguments to be fed to the constructor of the visualization
        class.
    """
    layout = core.SimpleLayout(wtitle=title)
    plot = layout.inject(DiffGanttVisualization, spskey="all", title=title)
    utils.bulksetattr(plot, **kwargs)
    plot.build(jobsets)
    layout.show()
