# coding: utf-8
from __future__ import unicode_literals, print_function
from collections import OrderedDict
import pandas as pd
import re
import datetime


class Workload(object):
    def __init__(self, swf_file=None):
        if swf_file:
            '''
             swf is the default format
             see: http://www.cs.huji.ac.il/labs/parallel/workload/swf.html

             the data format is one line per job, with 18 fields:
              0 - Job Number, a counter field, starting from 1
              1 - Submit Time, seconds. submittal time
              2 - Wait Time, seconds. diff between submit and begin to run
              3 - Run Time, seconds. end-time minus start-time
              4 - Number of Processors, number of allocated processors
              5 - Average CPU Time Used, seconds. user+system. avg over procs
              6 - Used Memory, KB. avg over procs.
              7 - Requested Number of Processors, requested number of processors
              8 - Requested Time, seconds. user runtime estimation
              9 - Requested Memory, KB. avg over procs.
             10 - status (1=completed, 0=killed), 0=fail; 1=completed; 5=canceled
             11 - User ID, user id
             12 - Group ID, group id
             13 - Executable (Application) Number, [1,2..n] n = app# appearing in log
             14 - Queue Number, [1,2..n] n = queue# in the system
             15 - Partition Number, [1,2..n] n = partition# in the systems
             16 - Preceding Job Number,  cur job will start only after ...
             17 - Think Time from Preceding Job, seconds should elapse between termination of
            '''
            columns = ['job', 'submit', 'wait', 'runtime', 'proc_alloc', 'cpu_used', 'mem_used', 'proc_req',
                       'user_est', 'mem_req', 'status', 'uid', 'gid', 'exe_num', 'queue', 'partition',
                       'prev_jobs', 'think_time']
            # columns_1 = ['Job', 'Submit Time', 'Wait Time', 'Run Time', 'Nb Alloc Proc', 'Average CPU',
            #             'Used Memory', 'Req Nb Proc', 'Req Time', 'Req Memory', 'Status', 'User',
            #             'Group', 'Application', 'Queue', 'Partition', 'Preceding Job', 'Think Time']

            self.df = pd.read_csv(swf_file, comment=';', names=columns, header=0, delim_whitespace=True)

            self.nb_jobs = len(self.df)
            # process header
            header_file = open(swf_file, 'r')
            self.header = ''
            for line in header_file:
                if re.match("^;", line):
                    print(line)
                    self.header += line

                    m = re.search(".*UnixStartTime:\s(\d+)", line)
                    if m:
                        self.unix_start_time = int(m.group(1))
                        self.str_start_time = datetime.datetime.fromtimestamp(
                            self.unix_start_time).strftime('%Y-%m-%d %H:%M:%S')

                    m = re.search(".*MaxNodes:\s(\d+)", line)
                    if m:
                        self.max_nodes = int(m.group(1))

                    m = re.search(".*MaxProcs:\s(\d+)", line)
                    if m:
                        self.max_procs = int(m.group(1))

                    m = re.search(".MaxQueues*:\s(\d+)", line)
                    if m:
                        self.max_queues = int(m.group(1))

                else:
                    # header is finished
                    break

    def utilisation(self):
        '''
        Calculate cluster utilisation over time:
        nb procs used / nb procs available
        '''
        df = self.df.sort_values(by='submit')
        df['start'] = df['submit'] + df['wait']
        df['stop'] = df['start'] + df['runtime']

        # Cleaning:
        # - still running jobs (runtime = -1)
        # - not scheduled jobs (wait = -1)
        # - no procs allocated (proc_alloc = -1)
        max_time = df['stop'].max()
        df.ix[df['runtime'] == -1, 'stop'] = max_time
        df.ix[df['runtime'] == -1, 'start'] = max_time
        df = df[df['proc_alloc'] > 0]

        # Create a list of start and stop event associated to the number of
        # proc allocation changes: starts add procs, stop remove procs
        event_columns = ['time', 'proc_alloc']
        start_event_df = pd.concat([df['start'],
                                    df['proc_alloc']],
                                   axis=1)
        start_event_df.columns = event_columns
        # Stop event give negative proc_alloc value
        stop_event_df = pd.concat([df['stop'],
                                   - df['proc_alloc']],
                                  axis=1)
        stop_event_df.columns = event_columns

        # merge events and sort them
        event_df = start_event_df.append(
            stop_event_df,
            ignore_index=True).sort_values(by='time').reset_index(drop=True)

        # convert timestamp to datetime
        event_df.index = event_df.apply(
            lambda x: datetime.datetime.fromtimestamp(
                self.unix_start_time + x['time']), axis=1)

        # merge events with the same timestamp
        return event_df.groupby(event_df.index).sum()['proc_alloc'].cumsum()

    def gene_arriving_each_day(self):
        df = self.df
        df['day'] = df.apply(lambda x: datetime.datetime.fromtimestamp(
            self.unix_start_time+x['submit']).strftime('%u'), axis=1)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        grouped = df.groupby('day')
        df1 = grouped['job'].agg({'jobs': lambda x: float(x.count())/self.nb_jobs})
        nb_procs = df['proc_req'].sum()
        df2 = grouped['proc_req'].agg({'procs': lambda x: float(x.sum())/nb_procs})

        df1['procs'] = df2['procs']
        df1.index = days
        self.arriving_each_day = df1

    def gene_arriving_each_hour(self):
        df = self.df
        df['hour'] = df.apply(lambda x: datetime.datetime.fromtimestamp(
            self.unix_start_time+x['submit']).strftime('%H'), axis=1)

        grouped = df.groupby('hour')
        df1 = grouped['job'].agg({'jobs': lambda x: float(x.count())/self.nb_jobs})
        nb_procs = df['proc_req'].sum()
        df2 = grouped['proc_req'].agg({'procs': lambda x: float(x.sum())/nb_procs})

        df1['procs'] = df2['procs']
        self.arriving_each_hour = df1

    def gene_jobs_per_week_per_user(self, nb=6):
        df = self.df
        df['week'] = df.apply(lambda x: datetime.datetime.fromtimestamp(
            self.unix_start_time+x['submit']).strftime('%W'), axis=1)
        grouped = df.groupby(['week', 'uid'])

        da = grouped['job'].agg({'count': 'count'})
        nb_weeks = len(da.index.levels[0])

        # identify nb largest contributors (uid), 0 uid is for others)
        job_nlargest_uid = list(df.groupby('uid')['job'].count().nlargest(nb).index)
        job_nlargest_uid_0 = [0] + job_nlargest_uid[::-1]  # [ 0, reversed list ]

        # jobs_week_uid_dict = {i: [0]*nb_weeks for i in job_nlargest_uid_0}
        jobs_week_uid_dict = OrderedDict((i, [0]*nb_weeks) for i in job_nlargest_uid_0)
        idx = 0
        for i, g in da.groupby(level=0):
            df_j = g.xs(i, level='week')  # dataframe uid / nb_jobs(count)
            for uid, row in df_j.iterrows():
                if uid in job_nlargest_uid_0:
                    print('uid:', uid)
                    jobs_week_uid_dict[uid][idx] = row['count']
                else:
                    jobs_week_uid_dict[0][idx] += row['count']
            idx += 1

        self.jobs_per_week_per_users = pd.DataFrame(jobs_week_uid_dict)

    def gene_procs_per_week_per_user(self):
        pass

    def gene_fraction_jobs_by_job_size(self):
        grouped = self.df.groupby('proc_alloc')
        self.fraction_jobs_by_job_size = grouped['job'].agg({'jobs': lambda x: float(x.count())/self.nb_jobs})
        
        # fig = plt.figure()
        # ax = fig.add_subplot(2, 1, 1)
        # ax.set_xscale('log', basex=2) need to bars' width
        # http://stackoverflow.com/questions/27534350/set-constant-width-to-every-bar-in-a-bar-plot
        # df1.plot(kind='bar', ax=ax)
        # 
    def gene_fraction_jobs_by_job_runtime(self):
        # grouped = self.df.groupby('runtime')
        # self.fraction_jobs_by_job_runtime = grouped['job']\
        #    .agg({'jobs': lambda x: float(x.count())/self.nb_jobs})

        # fig = plt.figure()
        # ax = fig.add_subplot(2, 1, 1)
        # ax.set_xscale('log'
        # df.hist(bins=np.logspace(1, 4.5 , 100), column='runtime')
        pass

    def generate_swf_log_dfs(self):
        '''Generate dataframes usable to plot swf log graphs'''
        print("Generate arriving_each_day dataframe")
        self.gene_arriving_each_day()
        print("Generate arriving_each_hour dataframe")
        self.gene_arriving_each_hour()
