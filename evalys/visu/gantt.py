# coding: utf-8

import functools

import matplotlib.dates
import matplotlib.patches
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


import numpy
import pandas
import os

from . import core
from .. import utils


def NOLABEL(_):
    """Labeler strategy disabling the labeling of jobs."""
    return ""


def _exitstate_legend():
    """
    :return Legend for the exitstate coloration method
    """
    return [Patch(facecolor="#35F500", edgecolor='black',
                  label='COMPLETED'),
            Patch(facecolor="#00FFEA", edgecolor='black',
                  label='TIMEOUT'),
            Patch(facecolor="#FF8604", edgecolor='black',
                  label='CANCELLED'),
            Patch(facecolor="#FF0000", edgecolor='black',
                  label='FAILED'),
            Patch(facecolor="#FFF700", edgecolor='black',
                  label='NODE_FAIL'),
            Patch(facecolor="#AE00FF", edgecolor='black',
                  label='RUNNING'),
            Patch(facecolor="#0019FF", edgecolor='black',
                  label='OUT_OF_MEMORY'),
            ]


def _sched_legend():
    """
    :return Legend for the scheduler job coloration method
    """
    return [Patch(facecolor="#7A0200", edgecolor='black',
                  label='SchedBackfill'),
            Patch(facecolor="#246A73", edgecolor='black',
                  label='SchedSubmit'),
    ]

def _wait_legend():
    """
    :return a legend for the wait-time job coloration method
    """
    return [
        Patch(facecolor="#95374F", edgecolor='black', alpha=1, label="Highest wait time"),
        Patch(facecolor="#95374F", edgecolor='black', alpha=0.75, label="High wait time"),
        Patch(facecolor="#95374F", edgecolor='black', alpha=0.5, label="Average wait time"),
        Patch(facecolor="#95374F", edgecolor='black', alpha=0.25, label="Low wait time"),
        Patch(facecolor="#95374F", edgecolor='black', alpha=0, label="Lowest wait time"),
    ]

def _power_legend():
    """
    :return a legend for the power job coloration method
    """
    return [
        Patch(facecolor=mcolors.to_rgba("#95374F", alpha=1.0), edgecolor='black', label="Highest power factor"),
        Patch(facecolor=mcolors.to_rgba("#95374F", alpha=0.75), edgecolor='black', label="High power factor"),
        Patch(facecolor=mcolors.to_rgba("#95374F", alpha=0.5), edgecolor='black', label="Average power factor"),
        Patch(facecolor=mcolors.to_rgba("#95374F", alpha=0.25), edgecolor='black', label="Low power factor"),
        Patch(facecolor=mcolors.to_rgba("#95374F", alpha=0), edgecolor='black', label="Lowest power factor"),
    ]

def _sched_border_legend():
    """
    :return: a legend for the scheduler border coloration of jobs
    """
    return [Patch(facecolor="white", edgecolor='#FF0400',
                  label='SchedBackfill'),
            Patch(facecolor="white", edgecolor='#00E1FF',
                  label='SchedSubmit'),
    ]


def _sched_edge_color(job):
    """
    :return: the edge color from a job based on what scheduler was used to schedule it
    """
    if "SchedBackfill" in job["flags"]:
        return "#FF0400"
    elif "SchedSubmit" in job["flags"]:
        return "#00E1FF"
    else:
        return "black"


def _default_edge_color(job):
    """
    :return: the default edge color of a job
    """
    return "black"


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
        self.alpha = 0.6
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
        """
        :return: a color to apply to :job based on :palette
        """
        return palette[job["uniq_num"] % len(palette)]

    @staticmethod
    def project_color_map(job, palette):
        """
        :return: a color to apply to the job based on the corresponding project
        """
        return palette[job["account"] - 1]

    @staticmethod
    def partition_color_map(job, palette):
        """
        :return: a color to apply to the job based on the corresponding partition
        """
        return palette[job["partition"]]

    @staticmethod
    def dependency_color_map(job, palette):
        """
        :return: a color to apply to the job based on job dependency chains
        """
        return palette[job["dependency_chain_head"] % len(palette)]

    @staticmethod
    def user_color_map(job, palette):
        """
        :return: a color to apply to the job based on the user who launched it
        """
        return palette[job["user"]]

    @staticmethod
    def top_user_color_map(job, palette):
        """
        :return: a color to apply to a job based on the user who launched it, if that user is in the top X percent of users
        """
        return palette[job["user_id"]]

    def _coloration_middleman(self, job, x0, duration, height, itv, colorationMethod="default", num_projects=None,
                              num_top_users=None, partition_count=0, num_users=None, edgeMethod="default"):
        """
        Taking into account the specified coloration method and edge coloration method, return the method to use to color jobs in the gantt chart.
        :param colorationMethod: the method to apply to the coloration of the fill of the job
        :param edgeMethod: the method to apply to the coloration of the edges of the job
        :return: A composite method function to be used to apply coloration to each job in the gantt chart
        """
        coloration_methods = {
            "default": self._return_default_rectangle,
            "project": self._return_project_rectangle if num_projects is not None else None,
            "dependency": self._return_dependency_rectangle,
            "user": self._return_user_rectangle if num_users is not None else None,
            "user_top_20": self._return_top_user_rectangle if num_top_users is not None else None,
            "sched": self._return_sched_rectangle,
            "wait": self._return_wait_rectangle,
            "partition": self._return_partition_rectangle,
            "exitstate": self._return_success_rectangle,
            "power": self._return_power_rectangle,

        }

        edge_coloration_methods = {
            "default": _default_edge_color,
            "sched": _sched_edge_color,
        }

        method_func = coloration_methods.get(colorationMethod, self._return_default_rectangle)
        edge_func = edge_coloration_methods.get(edgeMethod, _default_edge_color)
        edge_color = edge_func(job)
        return method_func(job, x0, duration, height, itv, num_projects, num_users, num_top_users, partition_count,
                           edge_color)


    def _return_default_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                  num_top_users=None, partition_count=None, edge_color="black"):
        return self._create_rectangle(job, x0, duration, height, itv, self.colorer, edge_color=edge_color), None

    def _return_project_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                  num_top_users=None, partition_count=None, edge_color="black"):
        palette_used = core.generate_palette(num_projects)
        return self._create_rectangle(job, x0, duration, height, itv, self.project_color_map, edge_color=edge_color,
                                      palette=palette_used), palette_used

    def _return_dependency_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                     num_top_users=None, partition_count=None, edge_color="black"):
        palette_used = core.generate_palette(8)
        return self._create_rectangle(job, x0, duration, height, itv, self.dependency_color_map,
                                      palette=palette_used, edge_color=edge_color), palette_used

    def _return_user_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                               num_top_users=None, partition_count=None, edge_color="black"):
        palette_used = core.generate_palette(num_users + 1)
        return self._create_rectangle(job, x0, duration, height, itv, self.user_color_map,
                                      palette=palette_used, edge_color=edge_color), palette_used

    def _return_top_user_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                   num_top_users=None, partition_count=None, edge_color="black"):
        if job["user_id"] != 0:
            palette_used = core.generate_palette(num_top_users)
            return self._create_rectangle(job, x0, duration, height, itv, self.top_user_color_map,
                                          palette=palette_used, edge_color=edge_color), palette_used
        else:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#C2C2C2", facecolor="#C2C2C2",
                                          edge_color=edge_color), None

    def _return_sched_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                num_top_users=None, partition_count=None, edge_color="black"):
        if "SchedBackfill" in job["flags"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#7A0200", facecolor="#7A0200",
                                          edge_color=edge_color), None
        elif "SchedSubmit" in job["flags"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#246A73", facecolor="#246A73",
                                          edge_color=edge_color), None
        else:
            return self._create_rectangle(job, x0, duration, height, itv, self.colorer, edge_color=edge_color), None

    def _return_success_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                  num_top_users=None, partition_count=None, edge_color="black"):
        edge_color="black"
        if "COMPLETED" in job["success"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#35F500", facecolor="#35F500",
                                          edge_color=edge_color), None

        elif "TIMEOUT" in job["success"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#00FFEA", facecolor="#00FFEA",
                                          edge_color=edge_color), None

        elif "CANCELLED" in job["success"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#FF8604", facecolor="#FF8604",
                                          edge_color=edge_color), None

        elif "FAILED" in job["success"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#FF0000", facecolor="#FF0000",
                                          edge_color=edge_color), None

        elif "NODE_FAIL" in job["success"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#FFF700", facecolor="#FFF000",
                                          edge_color=edge_color), None

        elif "RUNNING" in job["success"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#AE00FF", facecolor="#AE00FF",
                                          edge_color=edge_color), None

        elif "OUT_OF_MEMORY" in job["success"]:
            return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#0019FF", facecolor="#0019FF",
                                          edge_color=edge_color), None

    def _return_wait_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                               num_top_users=None, partition_count=None, edge_color="black"):
        return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#95374F", facecolor=mcolors.to_rgba("#95374F", alpha=job["normalized_eligible_wait"]), edge_color=edge_color, alphaOverride=True), None

    def _return_power_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                               num_top_users=None, partition_count=None, edge_color="black"):
        return self._create_rectangle(job, x0, duration, height, itv, lambda _: "#95374F", facecolor=mcolors.to_rgba("#95374F", alpha=job["normalizedPowerFactor"]), edge_color=edge_color, alphaOverride=True), None

    def _return_partition_rectangle(self, job, x0, duration, height, itv, num_projects=None, num_users=None,
                                    num_top_users=None, partition_count=None, edge_color="black"):
        palette_used = core.generate_palette(partition_count)
        return self._create_rectangle(job, x0, duration, height, itv, self.partition_color_map,
                                      alpha=job["normalized_account"], edge_color=edge_color,
                                      palette=palette_used), palette_used

    def _create_rectangle(self, job, x0, duration, height, itv, color_func, alpha=-1, edge_color="black", palette=None,
                          facecolor=None, alphaOverride=False):
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
            if not alphaOverride:
                return matplotlib.patches.Rectangle(
                    (x0, itv.inf),
                    duration,
                    height,
                    alpha=alpha,
                    facecolor=facecolor,
                    edgecolor=edge_color,
                    linewidth=0.5,
                )
            else:
                return matplotlib.patches.Rectangle(
                    (x0, itv.inf),
                    duration,
                    height,
                    facecolor=facecolor,
                    edgecolor=edge_color,
                    linewidth=0.5,
                )

    def _draw(
            self, df, resvStart=None, resvExecTime=None, resvNodes=None, resvSet=None, colorationMethod="default",
            num_projects=None, num_users=None, num_top_users=None, partition_count=0, edgeMethod="default"
    ):
        """
        Draw a Gantt chart containing all jobs that fit within the window
        """
        def _plot_job(job, colorationMethod="default", num_projects=None, num_top_users=None, partition_count=0,
                      edgeMethod="default"):
            """
            This function is used to plot each individual job
            """
            x0 = job["starting_time"]
            duration = job["execution_time"]
            try:
                if job["purpose"] != "reservation":
                    for itv in job["allocated_resources"].intervals():
                        height = itv.sup - itv.inf + 1
                        rect, palette_used = self._coloration_middleman(job, x0, duration, height, itv, colorationMethod, num_projects,
                                                          num_top_users, partition_count, num_users, edgeMethod)
                        self._ax.add_artist(rect)

                        if colorationMethod == "user" or colorationMethod == "user_top_20":
                            self._annotate(rect, str(job["username"]))
                        if colorationMethod == "dependency":
                            if job["dependency_chain_head"] != job["jobID"]:
                                self._annotate(rect, str(job["dependency_chain_head"]))
                        if colorationMethod == "project" or colorationMethod == "partition":
                            self._annotate(rect, job["account_name"])
            except:
                pass

                    # self._annotate(rect, self.labeler(job))

        df.apply(_plot_job, axis="columns", colorationMethod=colorationMethod, num_projects=num_projects,
                 partition_count=partition_count, num_top_users=num_top_users, edgeMethod=edgeMethod)

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
        """
        Builds a gantt chart from the provided :jobset
        """
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
            edgeMethod="default",
    ):
        """
        Build a Gantt chart from a provided DataFrame
        """
        column_mapping = {
            "project": self.COLUMNS + ("account", "account_name", "flags",),
            "dependency": self.COLUMNS + ("dependency_chain_head", "flags",),
            "user": self.COLUMNS + ("user", "username", "user_id", "flags",),
            "user_top_20": self.COLUMNS + ("user", "username", "user_id", "flags",),
            "sched": self.COLUMNS + ("flags",),
            "wait": self.COLUMNS + ("normalized_eligible_wait", "flags",),
            "partition": self.COLUMNS + ("partition", "account", "normalized_account", "account_name", "flags",),
            "exitstate": self.COLUMNS + ("success", "flags",),
            "power": self.COLUMNS + ("powerFactor",),
        }
        # TODO Is the ordering of the code below having twice as many cols as needed?
        if "flags" in df.head():
            df = df.loc[:, column_mapping.get(colorationMethod, self.COLUMNS + ("flags",))]
        else:
            df = df.loc[:, column_mapping.get(colorationMethod, self.COLUMNS)]

        # Calculate the 0.01 and 0.99 percentiles
        if 'powerFactor' in df.head():
            df['powerFactor'].fillna(0, inplace=True)

            percentile_001 = numpy.percentile(df['powerFactor'], 5)
            percentile_099 = numpy.percentile(df['powerFactor'], 95)

            # Normalize and round the values
            df['normalizedPowerFactor'] = (df['powerFactor'] - percentile_001) / (percentile_099 - percentile_001)
            df['normalizedPowerFactor'] = numpy.clip(df['normalizedPowerFactor'], 0, 1)  # Clip values to be between 0 and 1
            df['normalizedPowerFactor'] = df['normalizedPowerFactor'].round(2)  # Round to 2 decimal places

        self._adapt(df)  # extract the data required for the visualization
        self._customize_layout()  # prepare the layout for displaying the data
        self._draw(
            df, resvStart, resvExecTime, resvNodes, resvSet, colorationMethod, num_projects, num_users, num_top_users,
            partition_count, edgeMethod
        )  # do the painting job
        # My axis setting method
        self._ax.set(
            xlim=(windowStartTime, windowFinishTime),
            ylim=(res_bounds.inf - 1, res_bounds.sup + 2),
        )

        legend_mapping = {
            "exitstate": _exitstate_legend,
            "sched": _sched_legend,
            "wait": _wait_legend,
            "power": _power_legend,
        }

        legend = legend_mapping.get(colorationMethod)

        if legend:
            legend_func = legend_mapping.get(colorationMethod)
            legend_elements = legend_func()
            self._ax.legend(handles=legend_elements, loc="upper left")
        if not legend:
            edge_legend_mapping = {"sched": _sched_border_legend}
            legend = edge_legend_mapping.get(edgeMethod)
            if legend:
                legend_func = edge_legend_mapping.get(edgeMethod)
                legend_elements = legend_func()
                self._ax.legend(handles=legend_elements, loc="upper left")


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
        dimensions=(6.4, 4.8),
        colorationMethod="default",
        num_projects=None,
        num_users=None,
        num_top_users=None,
        partition_count=0,
        edgeMethod="default",
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
    layout = core.SimpleLayout(wtitle=title, dimensions=dimensions)
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
        edgeMethod,
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
