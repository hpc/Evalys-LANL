# coding: utf-8
from __future__ import unicode_literals, print_function
import numpy as np
import matplotlib.pyplot as plt
from evalys import visu
import evalys.visu.legacy as vleg
from procset import ProcInt, ProcSet
from evalys.metrics import compute_load, load_mean, fragmentation_reis, fragmentation
import warnings

# FIXME Fix this

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd

class JobSet(object):
    """
    A JobSet is a set of jobs with their state, their time properties and
    the resources they are associated to.

    It takes a dataframe as input that is intended to have the columns
    defined in :py::`JobSet.columns`.

    The `allocated_resources` one should contain the string representation
    of an interval set of the allocated resources for the given job, i.e.
    for this interval::

        # interval_set representation
        [(1, 2), (5, 5), (10, 50)]
        # strinf representation
        1-2 5 10-50

    .. warning:: Floating point precision is set to
        :py:attr:`self.float_precision` so all floating point values are
        rounded with this number of digits. Defalut set to 6

    For example:

    >>> from evalys.jobset import JobSet
    >>> js = JobSet.from_csv("./examples/jobs.csv")
    >>> js.plot(with_details=True)
    >>> # to show the graph
    >>> # import matplotlib.pyplot as plt
    >>> # plt.show()

    You can also specify the the resource_bounds like this:

    >>> js = JobSet.from_csv("./examples/jobs.csv",
    ...                      resource_bounds=(0, 63))
    """

    def __init__(self, df, resource_bounds=None, float_precision=6):
        # reset the index of the dataframe
        df = df.reset_index(drop=True)
        # set float round precision
        self.float_precision = float_precision
        self.df = np.round(df, float_precision)

        if resource_bounds:
            self.res_bounds = ProcInt(*resource_bounds)
        else:

            def alloc_apply(f, alloc):
                for pset in alloc:
                    try:
                        yield f(pset)
                    except ValueError:
                        pass

            self.res_bounds = ProcInt(
                min(alloc_apply(lambda pset: pset.min, self.df.allocated_resources)),
                max(alloc_apply(lambda pset: pset.max, self.df.allocated_resources)),
            )
        self.MaxProcs = len(self.res_bounds)

        self.df["proc_alloc"] = self.df.allocated_resources.apply(len)

        # Add missing columns if possible
        fillable_relative = all(
            col in self.df.columns
            for col in ["submission_time", "waiting_time", "execution_time"]
        )
        fillable_absolute = all(
            col in self.df.columns
            for col in ["submission_time", "starting_time", "finish_time"]
        )
        if fillable_relative:
            if "starting_time" not in self.df.columns:
                self.df["starting_time"] = (
                    self.df["submission_time"] + self.df["waiting_time"]
                )
            if "finish_time" not in self.df.columns:
                self.df["finish_time"] = (
                    self.df["starting_time"] + self.df["execution_time"]
                )
        elif fillable_absolute:
            if "waiting_time" not in self.df.columns:
                self.df["waiting_time"] = (
                    self.df["starting_time"] - self.df["submission_time"]
                )
            if "execution_time" not in self.df.columns:
                self.df["execution_time"] = (
                    self.df["finish_time"] - self.df["starting_time"]
                )

        if "job_id" in self.df.columns:
            self.df.rename(columns={"job_id": "jobID"}, inplace=True)

        # init cache
        self._utilisation = None
        self._queue = None

    __converters = {
        "jobID": str,
        "job_id": str,
        "workload": str,
        "profile": str,
        "allocated_resources": ProcSet.from_str,
    }

    columns = [
        "job_id",
        "workload_name",
        "submission_time",
        "requested_number_of_resources",
        "requested_time",
        "success",
        "starting_time",
        "execution_time",
        "finish_time",
        "waiting_time",
        "turnaround_time",
        "stretch",
        "allocated_resources",
    ]

    @classmethod
    def from_csv(cls, filename, resource_bounds=None):
        """
        Return a JobSet calculated from the provided CSV file.
        :param filename: Name of CSV file to parse
        :param resource_bounds: Resource bounds to apply to the JobSet. This can be correlated with Node IDs.
        :return: a JobSet with the DF loaded from the given CSV file and resource_bounds.
        """
        df = pd.read_csv(filename, converters=cls.__converters)
        return cls(df, resource_bounds=resource_bounds)

    @classmethod
    def from_df(cls, df, resource_bounds=None):
        """
        Return a JobSet calculated from the provided DataFrame.
        :param df: The DataFrame to use in the JobSet
        :param resource_bounds: Resource bounds to apply to the JobSet. This can be correlated with Node IDs.
        :return: a JobSet with the DF loaded from the given DF and resource_bounds.
        """
        return cls(df, resource_bounds=resource_bounds)

    def to_csv(self, filename):
        """Export this jobset to a csv file with a ',' as separator.

        Example:

        >>> from evalys.jobset import JobSet
        >>> js = JobSet.from_csv("./examples/jobs.csv")
        >>> js.to_csv("/tmp/jobs.csv")
        """
        df = self.df.copy()
        df.allocated_resources = df.allocated_resources.apply(str)
        with open(filename, "w") as f:
            df.to_csv(
                f,
                index=False,
                sep=",",
                float_format="%.{}f".format(self.float_precision),
            )

    def gantt(self, time_scale=False, **kwargs):
        """
        Quickly and simply plot a Gantt chart for the JobSet.
        """
        if time_scale:
            kwargs["xscale"] = "time"
        visu.plot_gantt(self, **kwargs)

    @property
    def utilisation(self):
        """
        Calculate the cluster utilization over time of the JobSet using metrics.compute_load
        :return:
        """
        if self._utilisation is not None:
            return self._utilisation
        self._utilisation = compute_load(
            self.df,
            col_begin="starting_time",
            col_end="finish_time",
            col_cumsum="proc_alloc",
        )
        return self._utilisation

    @property
    def queue(self):
        """
        Calculate cluster queue size over time in number of procs.

        :returns:
            a time indexed serie that contain the number of used processors
        """
        # Do not re-compute everytime
        if self._queue is not None:
            return self._queue

        proc = "requested_number_of_resources"
        self._queue = compute_load(self.df, "submission_time", "starting_time", proc)
        return self._queue

    def reset_time(self, to=0):
        """
        Reset the time index by giving the first submission time as 1
        """
        df = self.df
        if not to:
            reset_value = df["submission_time"].min() - 1
        else:
            reset_value = to
        for col in ["starting_time", "submission_time", "finish_time"]:
            df[col] = df[col] - reset_value

        self._queue = None
        self._utilisation = None

    def plot(
        self,
        longJs=None,
        largeJs=None,
        normalize=False,
        with_details=False,
        time_scale=False,
        title=None,
        with_gantt=False,
        reservationStart=None,
        reservationExec=None,
        reservationNodes=None,
        windowStartTime=None,
        windowFinishTime=None,
        binned=False,
        simple=False,
        timeline=False,
        clusterSize=None,
        count2=None,
    ):
        """
        Create a gantt chart from the JobSet with more specific control over the chart.
        :param longJs: Optional JobSet argument used in Binned chart type of BatsimGantt to contain jobs characterized as long
        :param largeJs: Optional JobSet argument used in Binned chart type of BatsimGantt to contain jobs characterized as large
        :param normalize: Optional boolean argument passed on to vleg.plot_load
        :param with_details: Optional boolean argument indicating whether to create a plot with additional job details
        :param time_scale: Optional boolean argument
        :param title: Optional argument indicating the title to apply to the plot
        :param with_gantt: Optional boolean argument that indicates whether a Gantt chart should be produced alongside the utilization plot
        :param reservationStart: Optional argument used in BatsimGantt to indicate reservation details
        :param reservationExec: Optional argument used in BatsimGantt to indicate reservation details
        :param reservationNodes: Optional argument used in BatsimGantt to indicate reservation details
        :param windowStartTime: Optional argument used by BatsimGantt and LiveGantt to indicate the start time of the window to chart
        :param windowFinishTime: Optional argument used by BatsimGantt and LiveGantt to indicate the finish time of the window to chart
        :param binned: Optional argument used by BatsimGantt to generate binned plots
        :param simple: Optional argument used to indicate to produce a simple chart
        :param timeline: Optional argument used by BatsimGantt to generate a timeline chart
        """
        nrows = 1
        if with_details and not binned:
            nrows = nrows + 2
        if (with_gantt) and not binned:
            nrows = nrows + 1
        if with_gantt and binned:
            nrows = nrows + 3
        fig, axe = plt.subplots(
            nrows=nrows, sharex=True, figsize=(12, 8)
        )  # FIXME I can override figsize here
        if title:
            fig.suptitle(title, fontsize=16)
        if ((not binned) or timeline) and not simple:
            try:
                # TODO What
                axeLen = len(axe)
                ax = axe[0]
            except TypeError:
                ax=axe

            # Choose the proper count of resources in order to accurately chart utilization for clusters with multiple non-contiguous blocks of nodes
            if count2 != None:
                resource_count = clusterSize + count2
            elif count2 == None and clusterSize != None:
                resource_count = clusterSize
            else:
                resource_count = self.MaxProcs

            if self.df["consumedEnergy"].unique().size <= 2:
                vleg.plot_load(
                    self.utilisation,
                    resource_count,
                    legend_label="utilisation",
                    ax=ax,
                    normalize=normalize,
                    time_scale=time_scale,
                    windowStartTime=windowStartTime,
                    windowFinishTime=windowFinishTime,
                )
            else:
                vleg.plot_load(
                    self.utilisation,
                    resource_count,
                    legend_label="utilisation",
                    ax=ax,
                    normalize=normalize,
                    time_scale=time_scale,
                    windowStartTime=windowStartTime,
                    windowFinishTime=windowFinishTime,
                    power=compute_load(
                        self.df,
                        col_begin="starting_time",
                        col_end="finish_time",
                        col_cumsum="consumedEnergy",
                    ),
                    normalize_power=False,
                )

        elif binned:
            fig.set_size_inches(30, 20)  # FIXME address this
            vleg.plot_binned_load(
                self.utilisation,
                longJs.utilisation,
                largeJs.utilisation,
                self.MaxProcs,
                legend_label="utilisation",
                ax=axe[0],
                normalize=normalize,
                time_scale=time_scale,
            )
        # vleg.plot_load(self.queue, self.MaxProcs,
        #                legend_label="queue", ax=axe[1], normalize=normalize,
        #                time_scale=time_scale)
        if with_details:
            vleg.plot_job_details(
                self.df, self.MaxProcs, ax=axe[1], time_scale=time_scale
            )
            vleg.plot_gantt(self, ax=axe[2], time_scale=time_scale)
        if with_gantt and not binned:
            vleg.plot_gantt(
                self,
                ax=axe[1],  # TODO Add title here
                time_scale=time_scale,
                labels=False,
                resvStart=reservationStart,
                resvExecTime=reservationExec,
                resvNodes=reservationNodes,
            )
        elif with_gantt and binned:
            vleg.plot_gantt(
                self,
                ax=axe[1],
                title="Small jobs",
                time_scale=time_scale,
                labels=False,
                resvStart=reservationStart,
                resvExecTime=reservationExec,
                resvNodes=reservationNodes,
                windowStartTime=windowStartTime,
                windowFinishTime=windowFinishTime,
            )
            vleg.plot_gantt(
                longJs,
                ax=axe[2],
                title="Long jobs",
                time_scale=time_scale,
                labels=False,
                resvStart=reservationStart,
                resvExecTime=reservationExec,
                resvNodes=reservationNodes,
                windowStartTime=windowStartTime,
                windowFinishTime=windowFinishTime,
            )
            vleg.plot_gantt(
                largeJs,
                ax=axe[3],
                title="Large Jobs",
                time_scale=time_scale,
                labels=False,
                resvStart=reservationStart,
                resvExecTime=reservationExec,
                resvNodes=reservationNodes,
                windowStartTime=windowStartTime,
                windowFinishTime=windowFinishTime,
            )

    def detailed_utilisation(self):
        # TODO I do not know what this does
        df = self.free_intervals()
        df["total"] = len(self.res_bounds) - df.free_itvs.apply(len)
        df.set_index("time", drop=True, inplace=True)
        return df

    def mean_utilisation(self, begin_time=None, end_time=None):
        return load_mean(self.utilisation, begin=begin_time, end=end_time)

    def free_intervals(self, begin_time=0, end_time=None):
        """
        :returns: a dataframe with the free resources over time. Each line
            corespounding to an event in the jobset.
        """
        df = self.df

        # Create a list of start and stop event associated to the proc
        # allocation:
        # Free -> Used : grab = 1
        # Used -> Free : grab = 0
        event_columns = ["time", "free_itvs", "grab"]
        start_event_df = pd.concat(
            [
                df["starting_time"],
                df["allocated_resources"],
                pd.Series(np.ones(len(df), dtype=bool)),
            ],
            axis=1,
        )
        start_event_df.columns = event_columns
        # Stop event have zero in grab
        stop_event_df = pd.concat(
            [
                df["finish_time"],
                df["allocated_resources"],
                pd.Series(np.zeros(len(df), dtype=bool)),
            ],
            axis=1,
        )
        stop_event_df.columns = event_columns

        # merge events and sort them
        event_df = pd.concat([start_event_df, stop_event_df],
            ignore_index=True).sort_values(
                by=['time', 'grab']).reset_index(drop=True)

        # cut events if necessary
        # reindex event_df
        event_df = event_df.sort_values(by="time").set_index(["time"], drop=False)
        # find closest index
        begin = event_df.index.searchsorted(begin_time)
        if end_time is not None:
            end = event_df.index.searchsorted(end_time)
        else:
            end = len(event_df.index) - 1

        event_df = event_df.iloc[begin:end].reset_index(drop=True)

        # All resources are free at the beginning
        event_columns = ["time", "free_itvs"]
        first_row = [begin_time, ProcSet(self.res_bounds)]
        free_interval_serie = pd.DataFrame(columns=event_columns)
        free_interval_serie.loc[0] = first_row
        for index, row in event_df.iterrows():
            current_itv = free_interval_serie.iloc[index]["free_itvs"]
            if row.grab:
                new_itv = current_itv - row.free_itvs
            else:
                new_itv = current_itv | row.free_itvs
            new_row = [row.time, new_itv]
            free_interval_serie.loc[index + 1] = new_row

        if end_time is not None:
            last_row = [end_time, ProcSet()]
            free_interval_serie.loc[len(free_interval_serie)] = last_row
        return free_interval_serie

    def free_slots(self, begin_time=0, end_time=None):
        """
        :returns: a DataFrame (compatible with a JobSet) that contains all
            the not overlapping square free slots of this JobSet maximzing the
            time. It can be transform to a JobSet to be plot as gantt chart.
        """
        # slots_time contains tuple of
        # (slot_begin_time,free_resources_intervals)
        free_interval_serie = self.free_intervals(begin_time, end_time)
        slots_time = [(free_interval_serie.time[0], ProcSet(self.res_bounds))]
        new_slots_time = slots_time
        columns = [
            "jobID",
            "allocated_resources",
            "starting_time",
            "finish_time",
            "execution_time",
            "submission_time",
        ]
        free_slots_df = pd.DataFrame(columns=columns)
        prev_free_itvs = ProcSet(self.res_bounds)
        slots = 0
        for i, curr_row in free_interval_serie.iterrows():
            if i == 0:
                continue
            new_slots_time = []
            curr_time = curr_row.time
            taken_resources = prev_free_itvs - curr_row.free_itvs
            freed_resources = curr_row.free_itvs - prev_free_itvs
            if i == len(free_interval_serie) - 1:
                taken_resources = ProcSet(self.res_bounds)
            if taken_resources:
                # slot ends: store it and update free slot
                for begin_time, itvs in slots_time:
                    to_update = itvs & taken_resources
                    if to_update:
                        # store new slots
                        slots = slots + 1
                        new_slot = [
                            str(slots),
                            to_update,
                            begin_time,
                            curr_time,
                            curr_time - begin_time,
                            begin_time,
                        ]
                        free_slots_df.loc[slots] = new_slot
                        # remove free slots
                        free_res = itvs - to_update
                        if free_res:
                            new_slots_time.append((begin_time, free_res))
                    else:
                        new_slots_time.append((begin_time, itvs))

            if freed_resources:
                # slots begin: udpate free slot
                if not new_slots_time:
                    new_slots_time = slots_time
                new_slots_time.append((curr_time, freed_resources))

            # update previous
            prev_free_itvs = curr_row.free_itvs
            # clean slots_free
            slots_time = new_slots_time
        return free_slots_df

    def fragmentation(
        self, p=2, resource_intervals=None, begin_time=None, end_time=None
    ):
        if end_time is None:
            end_time = self.df.finish_time.max()
        if begin_time is None:
            begin_time = self.df.submission_time.min()
        return fragmentation(
            self.free_resources_gaps(resource_intervals, begin_time, end_time), p=p
        )
        # return fragmentation_reis(
        #    self.free_resources_gaps(resource_intervals,
        #                             begin_time, end_time),
        #    end_time - begin_time, p=p)

    def free_resources_gaps(self, resource_intervals=None, begin_time=0, end_time=None):
        """
        :param resource_intervals: An interval set on which compute the
            free resources gaps, Default: self.res_bounds
        :returns: a resource indexed list where each element is a numpy
            array of free slots.
        """
        js = self
        fs = js.free_slots(begin_time, end_time)
        free_resources_gaps = []
        if resource_intervals is None:
            resource_intervals = self.res_bounds
        for _ in range(resource_intervals[0], resource_intervals[1] + 1):
            free_resources_gaps.append([])

        def get_free_slots_by_resources(x):
            for res in range(resource_intervals[0], resource_intervals[1] + 1):
                if res in x.allocated_resources:
                    free_resources_gaps[res - resource_intervals[0]].append(
                        x.execution_time
                    )

        # compute resource gaps
        fs.apply(get_free_slots_by_resources, axis=1)
        # format each gap list in numpy array
        for i, fi in enumerate(free_resources_gaps):
            free_resources_gaps[i] = np.asarray(fi)

        return free_resources_gaps
