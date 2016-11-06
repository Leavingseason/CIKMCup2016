using CIKM2016.Structure;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIKM2016.Models
{
    class Naive
    {
        public static Random rng = new Random((int)(DateTime.Now.Ticks));

        public static object locker = new object();

        public static DateTime _start_date = DateTime.Parse("2014-12-01");

        public static void GenTimeFeatures()
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features\train_timefeature_labeled00.csv";
            HashSet<string> visited_uid = new HashSet<string>(); 
 
            var user2fact = Loader.LoadUserFacts();
          
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");

            List<string> uid_set = user2fact.Keys.ToList();

            using (StreamReader rd = new StreamReader(infile))
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int cnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 1000 == 0)
                    {
                        Console.WriteLine("output pair {0}", cnt);
                    }
                    string[] words = content.Split(',');
                    TimeFeatureBag tf_ua = GenTimeFeature(user2fact[words[0]]);
                    TimeFeatureBag tf_ub = GenTimeFeature(user2fact[words[1]]);
                    if (user2fact.ContainsKey(words[0]) && user2fact.ContainsKey(words[1]) && user2fact[words[0]].facts.Count > 5 && user2fact[words[1]].facts.Count > 5)
                    {
                        WriteOneLine4TimeFeature(words[0], words[1], user2fact,wt, 1, tf_ua, tf_ub);
                    }

                    if (!visited_uid.Contains(words[0]))
                    {
                        SampleTimeFeature4OneUser(words[0], user2matches, user2fact, uid_set, wt, 10, tf_ua);
                        visited_uid.Add(words[0]);
                    }
                    if (!visited_uid.Contains(words[1]))
                    {
                        SampleTimeFeature4OneUser(words[1], user2matches, user2fact, uid_set, wt, 10, tf_ub);
                        visited_uid.Add(words[1]);
                    }
                     
                }
            }
        }

        private static TimeFeatureBag GenTimeFeature(Facts facts)
        {
            TimeFeatureBag tf = new TimeFeatureBag();
            List<DateTime> timelist = new List<DateTime>();
            foreach (var s in facts.facts)
            {
                timelist.Add(UserProfileInfer.Utils.Common.ParseTimeStampMillisecond(s.ts));
            }
            tf.firstdate = timelist[0];
            tf.lastdate = timelist[timelist.Count - 1];
            tf.hour2cnt = new double[24];
            tf.day2cnt = new double[7];
            tf.month2cnt = new double[20];
            tf.date2cnt = new Dictionary<string, int>();

            for (int i = 0; i < 24; i++)
            {
                tf.hour2cnt[i] = 0 ;
            }
            for (int i = 0; i < 20; i++)
            {
                tf.month2cnt[i]=0;
            }
            for (int i = 0; i < 7; i++)
            {
                tf.day2cnt[i]=0;
            }

            float unit = 1.0f / timelist.Count;

            foreach (var dt in timelist)
            {
                int houridx = dt.Hour;
                tf.hour2cnt[houridx] += unit;

                int day = (int)dt.DayOfWeek;
                tf.day2cnt[day] += unit;

                int month =( (int)dt.Subtract(_start_date).TotalDays )/ 30;
                tf.month2cnt[month] += unit;

                string datestr = dt.ToString("yyyy-MM-dd");
                if (!tf.date2cnt.ContainsKey(datestr))
                {
                    tf.date2cnt.Add(datestr, 1);
                }
                else
                {
                    tf.date2cnt[datestr]++;
                }
            }

            return tf;
        }

        private static void SampleTimeFeature4OneUser(string uid, Dictionary<string, HashSet<string>> user2matches, Dictionary<string, Facts> user2fact, List<string> uid_set, StreamWriter wt, int k  , TimeFeatureBag tf_ua)
        {
            for (int i = 0; i < k; i++)
            {
                string negid = SampleOneUid(uid, user2matches, uid_set, user2fact);
                if (negid == null)
                {
                    continue;
                }
                TimeFeatureBag tf_ub = GenTimeFeature(user2fact[negid]);
                WriteOneLine4TimeFeature(uid,negid,user2fact,wt, 0, tf_ua, tf_ub);
            }
        }

        public static string GenTimeFeatureLine(string uid, string negid, Dictionary<string, Facts> user2fact,   TimeFeatureBag tf_ua, TimeFeatureBag tf_ub)
        {
            double hourcor = Tools.MathLib.PearsonCorrelation(tf_ua.hour2cnt, tf_ub.hour2cnt);
            double daycor = Tools.MathLib.PearsonCorrelation(tf_ua.day2cnt, tf_ub.day2cnt);
            double monthcor = Tools.MathLib.PearsonCorrelation(tf_ua.month2cnt, tf_ub.month2cnt);

            double hourcross = Tools.MathLib.CrossEntropy(tf_ua.hour2cnt, tf_ub.hour2cnt);
            double daycross = Tools.MathLib.CrossEntropy(tf_ua.day2cnt, tf_ub.day2cnt);
            double monthcross = Tools.MathLib.CrossEntropy(tf_ua.month2cnt, tf_ub.month2cnt);

            double first_day_gap = Math.Abs(tf_ua.firstdate.Subtract(tf_ub.firstdate).TotalDays);
            double last_day_gap = Math.Abs(tf_ua.lastdate.Subtract(tf_ub.lastdate).TotalDays);

            int date_len = tf_ua.date2cnt.Count + tf_ub.date2cnt.Count;
            int com_date = 0;
            foreach (var datestr in tf_ua.date2cnt.Keys)
            {
                if (tf_ub.date2cnt.ContainsKey(datestr))
                {
                    com_date++;
                }
            }
            double overlap_dates = com_date * 2.0 / date_len;
            double heavy_ratio = user2fact[uid].facts.Count * 1.0 / user2fact[negid].facts.Count;
            if (heavy_ratio > 1)
            {
                heavy_ratio = 1.0 / heavy_ratio;
            }

            string res = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}", 
                 hourcor, daycor, monthcor, hourcross, daycross, monthcross, first_day_gap, last_day_gap, overlap_dates, heavy_ratio);

            return res;
        }

        private static void WriteOneLine4TimeFeature(string uid, string negid, Dictionary<string, Facts> user2fact, StreamWriter wt, int label,TimeFeatureBag tf_ua,TimeFeatureBag tf_ub)
        {
            string line = GenTimeFeatureLine(uid, negid, user2fact, tf_ua, tf_ub);

            wt.WriteLine("{0},{1},{2},{3}", label, uid, negid, line);

        }

        public static void GenTFIDF()
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\titles.csv";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\word_freq_title.csv";
            Dictionary<string, int> word2freq = new Dictionary<string, int>();
            using (StreamReader rd = new StreamReader(infile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    HashSet<string> tokens = new HashSet<string>(words[1].Split(' '));
                    foreach (var token in tokens)
                    {
                        if (!word2freq.ContainsKey(token))
                        {
                            word2freq.Add(token, 1);
                        }
                        else
                        {
                            word2freq[token]++;
                        }
                    }
                }
            }

            OutputSortedDict(word2freq, outfile);
        }

        public static void OutputSortedDict(Dictionary<string, int> dict, string outfile)
        {
            List<Tuple<string, int>> list = new List<Tuple<string, int>>();
            foreach (var pair in dict)
            {
                list.Add(new Tuple<string, int>(pair.Key, pair.Value));
            }
            list.Sort((a, b) => b.Item2.CompareTo(a.Item2));
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                foreach (var tuple in list)
                {
                    wt.WriteLine("{0},{1}", tuple.Item1, tuple.Item2);
                }
            }
        }

        public static void OutputSortedDict(Dictionary<string, double> dict, string outfile)
        {
            List<Tuple<string, double>> list = new List<Tuple<string, double>>();
            foreach (var pair in dict)
            {
                list.Add(new Tuple<string, double>(pair.Key, pair.Value));
            }
            list.Sort((a, b) => b.Item2.CompareTo(a.Item2));
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                foreach (var tuple in list)
                {
                    wt.WriteLine("{0},{1}", tuple.Item1, tuple.Item2);
                }
            }
        }

        /*
        public static void GenTrainCandidateWithLinearModel(int run_idx, int task_cnt)
        {
            Console.WriteLine(run_idx + "\t" + task_cnt);

            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features02\train_by_LR\train_candi" + run_idx + ".csv";
            var fid2usercnt = Loader.LoadFid2Usercnt();
            var url2freqcnt00 = Loader.LoadUrl2factcnt00();
            var url2freqcnt01 = Loader.LoadUrl2factcnt01();
            var url2freqcnt02 = Loader.LoadUrl2factcnt02();
            var url2freqcnt03 = Loader.LoadUrl2factcnt03();
            var word2doccnt = Loader.LoadWord2Doccnt();
            
            var fid2url = Loader.LoadFid2Url();
            var fid2words = Loader.LoadFid2Words();
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");

            var user2fact = Loader.LoadUserFacts(user2matches);

            List<string> uid_set = user2matches.Keys.ToList();


            Dictionary<string, Dictionary<string, int>> user2wordcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2fidcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2urlcnt = new Dictionary<string, Dictionary<string, int>>();
            foreach (var uid in uid_set)
            {
                Dictionary<string, int> fidcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> urlcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> wordcnt_usera = GetWordCnt(user2fact[uid], fid2words, fidcnt_usera, urlcnt_usera, fid2url, word2doccnt);
                user2wordcnt.Add(uid, wordcnt_usera);
                user2fidcnt.Add(uid, fidcnt_usera);
                user2urlcnt.Add(uid, urlcnt_usera);
            }


            int len = uid_set.Count;
            int finishedCnt = 0;

            int skip_cnt = 0;
            int search_limit = 60000;
            List<string> can_userset = new List<string>();
            for (int i = skip_cnt; i < len && i < skip_cnt+search_limit; i++)
            {
                if (i % task_cnt == run_idx)
                {
                    can_userset.Add(uid_set[i]);
                }
            }

            int clen = can_userset.Count;

            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int cnt = 0;
                ParallelOptions op = new ParallelOptions();
                op.MaxDegreeOfParallelism = 10;
                Parallel.For(0, clen, op, (i) =>
                {
                    //for (int i = 0; i < len; i++)

                    string uida = can_userset[i];
                    if (user2fact.ContainsKey(uida) && user2fact[uida].facts.Count > 5)
                    {

                        if (cnt++ % 1000 == 0)
                        {
                            Console.WriteLine("output pair {0}", cnt);
                        } 

                        List<Tuple<string, double>> curlist = new List<Tuple<string, double>>();

                        for (int j = 0; j < len; j++)
                        {
                            string uidb = uid_set[j];

                            if (user2matches[uida].Contains(uidb) || uidb.CompareTo(uida) <= 0 || !user2fact.ContainsKey(uidb) || user2fact[uidb].facts.Count <= 5)
                            {
                                continue;
                            }  

                            double wordsim = CalcTFIDF(user2wordcnt[uida], user2wordcnt[uidb], word2doccnt, 300125);
                            double fidsim = CalcTFIDF(user2fidcnt[uida], user2fidcnt[uidb], fid2usercnt, 300000);
                            double url00sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt00, 4066204);
                            if (url00sim <= 0)
                                continue;

                            double url01sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt01, 701900);
                            double url02sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt02, 154866);
                            double url03sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt03, 104866);
                            
                            double score = 1.779785 * wordsim + 7.905555 * fidsim - 0.7300617 * url00sim + 9.6302 * url01sim + 8.900364 * url02sim - 6.665703 * url03sim - 5.203684;
                            curlist.Add(new Tuple<string, double>(uid_set[j], score));
                        }

                        int topk = 90;
                        curlist.Sort((a, b) => b.Item2.CompareTo(a.Item2));

                        lock (locker)
                        {
                            for (int k = 0; k < topk && k < curlist.Count; k++)
                            {
                                wt.WriteLine("{0},{1},{2},{3}", 0, uida, curlist[k].Item1, 1.0 / (1 + Math.Exp(-1.0 * curlist[k].Item2)));
                            }
                            wt.Flush();
                        }  

                        foreach (var matchuid in user2matches[uida])
                        {
                            if (matchuid.CompareTo(uida) <= 0 || !user2fact.ContainsKey(matchuid) || user2fact[matchuid].facts.Count <= 5)
                            {
                                continue;
                            }

                            string uidb = matchuid;
                            double wordsim = CalcTFIDF(user2wordcnt[uida], user2wordcnt[uidb], word2doccnt, 300125);
                            double fidsim = CalcTFIDF(user2fidcnt[uida], user2fidcnt[uidb], fid2usercnt, 300000);
                            double url00sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt00, 4066204);
                            double url01sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt01, 701900);
                            double url02sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt02, 154866);
                            double url03sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt03, 104866);
                            if (url00sim <= 0)
                                continue;
                            double score = 1.779785 * wordsim + 7.905555 * fidsim - 0.7300617 * url00sim + 9.6302 * url01sim + 8.900364 * url02sim - 6.665703 * url03sim - 5.203684;
                            lock (locker)
                            {
                                wt.WriteLine("{0},{1},{2},{3}", 1, uida, uidb, 1.0 / (1 + Math.Exp(-1.0 * score)));
                            }
                        }

                        finishedCnt++;
                        Console.WriteLine("Finish cnt : {0}", finishedCnt);
                    }
                });
            }
        }
        */

        public static void GenTestCandidateWithLinearModel(int run_idx, int task_cnt)
        {
            Console.WriteLine(run_idx + "\t" + task_cnt);

            string outfile = @"E:\\new_test_candi_selection\test_candi" + run_idx + ".csv";

            var fid2usercnt = Loader.LoadFid2Usercnt();
            var url2freqcnt00 = Loader.LoadUrl2factcnt00();
            var url2freqcnt01 = Loader.LoadUrl2factcnt01();
            var url2freqcnt02 = Loader.LoadUrl2factcnt02();
            var url2freqcnt03 = Loader.LoadUrl2factcnt03();
            var word2doccnt = Loader.LoadWord2Doccnt();

            var fid2url = Loader.LoadFid2Url();
            var fid2words = Loader.LoadFid2Words();
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");
            var user2fact = Loader.LoadUserFacts();

            List<string> uid_set = new List<string>();
            foreach (var uid in user2fact.Keys)
            {
                if (!user2matches.ContainsKey(uid))
                    uid_set.Add(uid);
            }

            Console.WriteLine("calculating time features...");
            Dictionary<string, TimeFeatureBag> user2timefeature = new Dictionary<string, TimeFeatureBag>();
            foreach (var uid in uid_set)
            {               
                user2timefeature.Add(uid, GenTimeFeature(user2fact[uid]));
            }
            Console.WriteLine("calculating time features finished.");


            Console.WriteLine("calculating user data cnt...");
            Dictionary<string, Dictionary<string, int>> user2wordcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2fidcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2urlcnt = new Dictionary<string, Dictionary<string, int>>();
            foreach (var uid in uid_set)
            {
                Dictionary<string, int> fidcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> urlcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> wordcnt_usera = GetWordCnt(user2fact[uid], fid2words, fidcnt_usera, urlcnt_usera, fid2url, word2doccnt);
                user2wordcnt.Add(uid, wordcnt_usera);
                user2fidcnt.Add(uid, fidcnt_usera);
                user2urlcnt.Add(uid, urlcnt_usera);
            }
            Console.WriteLine("calculating user data cnt finished.");


            int len = uid_set.Count;
            int finishedCnt = 0;

            List<string> can_userset = new List<string>();
            for (int i = 0; i < len; i++)
            {
                if (i % task_cnt == run_idx)
                {
                    can_userset.Add(uid_set[i]);
                }
            }

            int clen = can_userset.Count;

            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int cnt = 0;
                ParallelOptions op = new ParallelOptions();
                op.MaxDegreeOfParallelism = 10;
                Parallel.For(0, clen, op, (i) =>
                {
                    //for (int i = 0; i < len; i++)

                    if ( user2fact[can_userset[i]].facts.Count > 5)
                    {

                        if (cnt++ % 1000 == 0)
                        {
                            Console.WriteLine("output pair {0}", cnt);
                        }
                        

                        List<Tuple<string, double>> curlist = new List<Tuple<string, double>>();

                        TimeFeatureBag tf_ua = null;
                        TimeFeatureBag tf_ub = null;
                        Dictionary<string, int> fidcnt_usera = null;
                        Dictionary<string, int> urlcnt_usera = null;
                        Dictionary<string, int> wordcnt_usera = null;

                        Dictionary<string, int> fidcnt_userb = null;
                        Dictionary<string, int> urlcnt_userb = null;
                        Dictionary<string, int> wordcnt_userb = null;

                        for (int j = 0; j < len; j++)
                        {
                            string uida = can_userset[i];
                            string uidb = uid_set[j];

                            if (uidb.CompareTo(uida) <= 0 || user2fact[uidb].facts.Count <= 5)
                            {
                                continue;
                            }
                             

                            if (user2fact.ContainsKey(uidb) && user2fact[uidb].facts.Count>5)
                            {
                                 tf_ua = user2timefeature[uida];
                                 tf_ub = user2timefeature[uidb];

                                 fidcnt_usera = user2fidcnt[uida];
                                 urlcnt_usera = user2urlcnt[uida];
                                 wordcnt_usera = user2wordcnt[uida];

                                 fidcnt_userb = user2fidcnt[uidb];
                                urlcnt_userb = user2urlcnt[uidb];
                                 wordcnt_userb = user2wordcnt[uidb];

                                double wordsim = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125);
                                double fidsim = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000);
                                double url00sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204);
                                double url01sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900);
                                double url02sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866);
                                double url03sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866);

                                string[] tfeaturelines = GenTimeFeatureLine(uida, uidb, user2fact, tf_ua, tf_ub).Split(',');
                                double[] timefeatures = new double[tfeaturelines.Length];
                                for (int g = 0; g < tfeaturelines.Length; g++)
                                {
                                    timefeatures[g] = double.Parse(tfeaturelines[g]);
                                }


                                //--
                                double wordsim02 = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125, 2);
                                double fidsim02 = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000, 2);
                                double url00sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204, 2);
                                double url01sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900, 2);
                                double url02sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866, 2);
                                double url03sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866, 2);

                                double wordsim03 = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125, 10, 50);
                                double fidsim03 = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000, 10, 50);
                                double url00sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204, 10, 50);
                                double url01sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900, 10, 50);
                                double url02sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866, 10, 50);
                                double url03sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866, 10, 50);

                                int comFidCnt = 0;
                                int fidSum = fidcnt_usera.Count + fidcnt_userb.Count;
                                foreach (var pair in fidcnt_usera)
                                {
                                    if (pair.Value > 1 && fidcnt_userb.ContainsKey(pair.Key))
                                    {
                                        comFidCnt++;
                                    }
                                }
                                double comFidRatio = comFidCnt * 1.0 / fidSum;

                                int comUrlCnt = 0;
                                int urlSum = urlcnt_usera.Count + urlcnt_userb.Count;
                                foreach (var pair in urlcnt_usera)
                                {
                                    if (pair.Value > 1 && urlcnt_userb.ContainsKey(pair.Key))
                                    {
                                        comUrlCnt++;
                                    }
                                }
                                double comUrlRatio = comUrlCnt * 1.0 / urlSum;

                                double score = FT100Pred(
                                     wordsim, fidsim, url00sim, url01sim, url02sim, url03sim
                                    , timefeatures[0], timefeatures[1], timefeatures[2], timefeatures[3], timefeatures[4], timefeatures[5], timefeatures[6], timefeatures[7], timefeatures[8], timefeatures[9]
                                    , Math.Min(user2fact[uida].facts.Count, user2fact[uidb].facts.Count)
                                    , wordsim02, fidsim02, url00sim02, url01sim02, url02sim02, url03sim02
                                    , wordsim03, fidsim03, url00sim03, url01sim03, url02sim03, url03sim03
                                    , comFidCnt, comFidRatio, comUrlCnt, comUrlRatio
                                );

                                curlist.Add(new Tuple<string, double>(uid_set[j], score));
                            }
                        }

                        int topk = 80;
                        curlist.Sort((a, b) => b.Item2.CompareTo(a.Item2));

                        lock (locker)
                        {
                            for (int k = 0; k < topk && k < curlist.Count; k++)
                            {
                                wt.WriteLine("{0},{1},{2}", can_userset[i], curlist[k].Item1, curlist[k].Item2);
                            }
                            wt.Flush();
                        }

                        finishedCnt++;
                        Console.WriteLine("Finish cnt : {0}", finishedCnt);
                    }
                });
            }
        }

        public static void GenTrainCandidateWithTreeModel(int run_idx, int task_cnt)
        {
            Console.WriteLine(run_idx + "\t" + task_cnt);

            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_tree\train_candi_report_LR_completelist" + run_idx + ".csv";

            var fid2usercnt = Loader.LoadFid2Usercnt();
            var url2freqcnt00 = Loader.LoadUrl2factcnt00();
            var url2freqcnt01 = Loader.LoadUrl2factcnt01();
            var url2freqcnt02 = Loader.LoadUrl2factcnt02();
            var url2freqcnt03 = Loader.LoadUrl2factcnt03();
            var word2doccnt = Loader.LoadWord2Doccnt();

            var fid2url = Loader.LoadFid2Url();
            var fid2words = Loader.LoadFid2Words();
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");
            var user2fact = Loader.LoadUserFacts(user2matches);

            List<string> uid_set = new List<string>(user2matches.Keys); 
             

            Console.WriteLine("calculating time features...");
            Dictionary<string, TimeFeatureBag> user2timefeature = new Dictionary<string, TimeFeatureBag>();
            foreach (var uid in uid_set)
            {
                user2timefeature.Add(uid, GenTimeFeature(user2fact[uid]));
            }
            Console.WriteLine("calculating time features finished.");


            Console.WriteLine("calculating user data cnt...");
            Dictionary<string, Dictionary<string, int>> user2wordcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2fidcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2urlcnt = new Dictionary<string, Dictionary<string, int>>();
            foreach (var uid in uid_set)
            {
                Dictionary<string, int> fidcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> urlcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> wordcnt_usera = GetWordCnt(user2fact[uid], fid2words, fidcnt_usera, urlcnt_usera, fid2url, word2doccnt);
                user2wordcnt.Add(uid, wordcnt_usera);
                user2fidcnt.Add(uid, fidcnt_usera);
                user2urlcnt.Add(uid, urlcnt_usera);
            }
            Console.WriteLine("calculating user data cnt finished.");


            int len = uid_set.Count;
            int finishedCnt = 0;

            int skipcnt = 60000;
            List<string> can_userset = new List<string>();
            //for (int i = skipcnt; i < len; i++)
            //{
            //    if (i % task_cnt == run_idx)
            //    {
            //        can_userset.Add(uid_set[i]);
            //    }
            //}
            string candifile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_LR\for_valid\valid_rng3000_merge_1000";
            using (StreamReader rd = new StreamReader(candifile))
            {
                string content = null;
                int i = 0;
                while ((content = rd.ReadLine()) != null)
                {
                    if (i++ % task_cnt == run_idx)
                        can_userset.Add(content);
                }
            }

            int clen = can_userset.Count;

            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int cnt = 0;
                ParallelOptions op = new ParallelOptions();
                op.MaxDegreeOfParallelism = 10;
                Parallel.For(0, clen, op, (i) =>
                {
                    //for (int i = 0; i < len; i++)

                    if (user2fact[can_userset[i]].facts.Count > 5)
                    {

                        if (cnt++ % 1000 == 0)
                        {
                            Console.WriteLine("output pair {0}", cnt);
                        }


                        List<Tuple<string, double>> curlist = new List<Tuple<string, double>>();

                        TimeFeatureBag tf_ua = null;
                        TimeFeatureBag tf_ub = null;
                        Dictionary<string, int> fidcnt_usera = null;
                        Dictionary<string, int> urlcnt_usera = null;
                        Dictionary<string, int> wordcnt_usera = null;

                        Dictionary<string, int> fidcnt_userb = null;
                        Dictionary<string, int> urlcnt_userb = null;
                        Dictionary<string, int> wordcnt_userb = null;

                        for (int j = 0; j < len; j++)
                        {
                            string uida = can_userset[i];
                            string uidb = uid_set[j];

                            if (uidb.CompareTo(uida) <= 0 || user2fact[uidb].facts.Count <= 5)
                            {
                                continue;
                            }


                            if (user2fact.ContainsKey(uidb) && user2fact[uidb].facts.Count > 5)
                            {
                                tf_ua = user2timefeature[uida];
                                tf_ub = user2timefeature[uidb];

                                fidcnt_usera = user2fidcnt[uida];
                                urlcnt_usera = user2urlcnt[uida];
                                wordcnt_usera = user2wordcnt[uida];

                                fidcnt_userb = user2fidcnt[uidb];
                                urlcnt_userb = user2urlcnt[uidb];
                                wordcnt_userb = user2wordcnt[uidb];

                                double wordsim = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125);
                                double fidsim = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000);
                                double url00sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204);
                                double url01sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900);
                                double url02sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866);
                                double url03sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866);

                                string[] tfeaturelines = GenTimeFeatureLine(uida, uidb, user2fact, tf_ua, tf_ub).Split(',');
                                double[] timefeatures = new double[tfeaturelines.Length];
                                for (int g = 0; g < tfeaturelines.Length; g++)
                                {
                                    timefeatures[g] = double.Parse(tfeaturelines[g]);
                                }


                                //--
                                double wordsim02 = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125, 2);
                                double fidsim02 = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000, 2);
                                double url00sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204, 2);
                                double url01sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900, 2);
                                double url02sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866, 2);
                                double url03sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866, 2);

                                double wordsim03 = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125, 10, 50);
                                double fidsim03 = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000, 10, 50);
                                double url00sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204, 10, 50);
                                double url01sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900, 10, 50);
                                double url02sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866, 10, 50);
                                double url03sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866, 10, 50);

                                int comFidCnt = 0;
                                int fidSum = fidcnt_usera.Count + fidcnt_userb.Count;
                                foreach (var pair in fidcnt_usera)
                                {
                                    if (pair.Value > 1 && fidcnt_userb.ContainsKey(pair.Key))
                                    {
                                        comFidCnt++;
                                    }
                                }
                                double comFidRatio = comFidCnt * 1.0 / fidSum;

                                int comUrlCnt = 0;
                                int urlSum = urlcnt_usera.Count + urlcnt_userb.Count;
                                foreach (var pair in urlcnt_usera)
                                {
                                    if (pair.Value > 1 && urlcnt_userb.ContainsKey(pair.Key))
                                    {
                                        comUrlCnt++;
                                    }
                                }
                                double comUrlRatio = comUrlCnt * 1.0 / urlSum;

                                //double score = FT100Pred(
                                //     wordsim, fidsim, url00sim, url01sim, url02sim, url03sim
                                //    , timefeatures[0], timefeatures[1], timefeatures[2], timefeatures[3], timefeatures[4], timefeatures[5], timefeatures[6], timefeatures[7], timefeatures[8], timefeatures[9]
                                //    , Math.Min(user2fact[uida].facts.Count, user2fact[uidb].facts.Count)
                                //    , wordsim02, fidsim02, url00sim02, url01sim02, url02sim02, url03sim02
                                //    , wordsim03, fidsim03, url00sim03, url01sim03, url02sim03, url03sim03
                                //    , comFidCnt, comFidRatio, comUrlCnt, comUrlRatio
                                //);

                                double score = 1.779785 * wordsim + 7.905555 * fidsim - 0.7300617 * url00sim + 9.6302 * url01sim + 8.900364 * url02sim - 6.665703 * url03sim - 5.203684;

                                curlist.Add(new Tuple<string, double>(uid_set[j], score));
                            }
                        }

                        int topk = curlist.Count; //80
                        //curlist.Sort((a, b) => b.Item2.CompareTo(a.Item2));

                        lock (locker)
                        {
                            for (int k = 0; k < topk && k < curlist.Count; k++)
                            {
                                wt.WriteLine("{0},{1},{2},{3}", user2matches[can_userset[i]].Contains(curlist[k].Item1) ? 1 : 0, can_userset[i], curlist[k].Item1, curlist[k].Item2);
                            }
                            wt.Flush();
                        }

                        finishedCnt++;
                        Console.WriteLine("Finish cnt : {0}", finishedCnt);
                    }
                });
            }
        }


        public static void Predict()
        {

            string[] words = "0.903980047724108,1.22557601665262,0.292571590571189,0.529846811509881,0.29311463741142,0.123449413436398".Split(',');
            double[] fes = new double[words.Length];
            for (int i = 0; i < fes.Length; i++)
            {
                fes[i] = double.Parse(words[i]);
            }
            double score = 1.779785 * fes[0] + 7.905555 * fes[1] - 0.7300617 * fes[2] + 9.6302 * fes[3] + 8.900364 * fes[4] - 6.665703 * fes[5] - 5.203684;
            Console.WriteLine(score);
            Console.WriteLine(1.0 / (1 + Math.Exp(-1.0 * score)));
        }

        /*
         public static void GenTestFileWithLinearModel(int run_idx, int task_cnt)
        {
            Console.WriteLine(run_idx + "\t" + task_cnt);

            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features02\train_by_LR\train_candi" + run_idx + ".csv";
            var fid2usercnt = Loader.LoadFid2Usercnt();
            var url2freqcnt00 = Loader.LoadUrl2factcnt00();
            var url2freqcnt01 = Loader.LoadUrl2factcnt01();
            var url2freqcnt02 = Loader.LoadUrl2factcnt02();
            var url2freqcnt03 = Loader.LoadUrl2factcnt03();
            var word2doccnt = Loader.LoadWord2Doccnt();
            var user2fact = Loader.LoadUserFacts();
            var fid2url = Loader.LoadFid2Url();
            var fid2words = Loader.LoadFid2Words();
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");



            List<string> uid_set = user2fact.Keys.ToList();


            Dictionary<string, Dictionary<string, int>> user2wordcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2fidcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2urlcnt = new Dictionary<string, Dictionary<string, int>>();
            foreach (var uid in uid_set)
            {
                Dictionary<string, int> fidcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> urlcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> wordcnt_usera = GetWordCnt(user2fact[uid], fid2words, fidcnt_usera, urlcnt_usera, fid2url, word2doccnt);
                user2wordcnt.Add(uid, wordcnt_usera);
                user2fidcnt.Add(uid, fidcnt_usera);
                user2urlcnt.Add(uid, urlcnt_usera);
            }


            int len = uid_set.Count;
            int finishedCnt = 0;

            List<string> can_userset = new List<string>();
            for (int i = 0; i < len; i++)
            {
                if (i % task_cnt == run_idx)
                {
                    can_userset.Add(uid_set[i]);
                }
            }

            int clen = can_userset.Count;

            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int cnt = 0;
                ParallelOptions op = new ParallelOptions();
                op.MaxDegreeOfParallelism = 10;
                Parallel.For(0, clen, op, (i) =>
                {
                    //for (int i = 0; i < len; i++)

                    if (!user2matches.ContainsKey(can_userset[i]) && user2fact[can_userset[i]].facts.Count > 5)
                    {

                        if (cnt++ % 1000 == 0)
                        {
                            Console.WriteLine("output pair {0}", cnt);
                        }
                        //Dictionary<string, int> fidcnt_usera = new Dictionary<string, int>();
                        //Dictionary<string, int> urlcnt_usera = new Dictionary<string, int>();
                        //Dictionary<string, int> wordcnt_usera = GetWordCnt(user2fact[uid_set[i]], fid2words, fidcnt_usera, urlcnt_usera, fid2url);


                        List<Tuple<string, double>> curlist = new List<Tuple<string, double>>();

                        for (int j = 0; j < len; j++)
                        {

                            if (user2matches.ContainsKey(uid_set[j]) || uid_set[j].CompareTo(can_userset[i]) <= 0 || user2fact[uid_set[j]].facts.Count <= 5)
                            {
                                continue;
                            }

                            string uida = can_userset[i];
                            string uidb = uid_set[j];

                            if (user2fact.ContainsKey(uidb))
                            {

                                //Dictionary<string, int> fidcnt_userb = new Dictionary<string, int>();
                                //Dictionary<string, int> urlcnt_userb = new Dictionary<string, int>();
                                //Dictionary<string, int> wordcnt_userb = GetWordCnt(user2fact[uid_set[j]], fid2words, fidcnt_userb, urlcnt_userb, fid2url);

                                //double wordsim = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125);
                                //double fidsim = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000);
                                //double url00sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204);
                                //double url01sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900);
                                //double url02sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866);
                                //double url03sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866);
                                double wordsim = CalcTFIDF(user2wordcnt[uida], user2wordcnt[uidb], word2doccnt, 300125);
                                double fidsim = CalcTFIDF(user2fidcnt[uida], user2fidcnt[uidb], fid2usercnt, 300000);
                                double url00sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt00, 4066204);
                                double url01sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt01, 701900);
                                double url02sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt02, 154866);
                                double url03sim = CalcTFIDF(user2urlcnt[uida], user2urlcnt[uidb], url2freqcnt03, 104866);
                                if (url00sim <= 0)
                                    continue;
                                double score = 1.779785 * wordsim + 7.905555 * fidsim - 0.7300617 * url00sim + 9.6302 * url01sim + 8.900364 * url02sim - 6.665703 * url03sim - 5.203684;
                                curlist.Add(new Tuple<string, double>(uid_set[j], score));
                            }
                        }

                        int topk = 100;
                        curlist.Sort((a, b) => b.Item2.CompareTo(a.Item2));

                        lock (locker)
                        {
                            for (int k = 0; k < topk && k < curlist.Count; k++)
                            {
                                wt.WriteLine("{0},{1},{2}", can_userset[i], curlist[k].Item1, 1.0 / (1 + Math.Exp(-1.0 * curlist[k].Item2)));
                            }
                            wt.Flush();
                        }

                        finishedCnt++;
                        Console.WriteLine("Finish cnt : {0}", finishedCnt);
                    }
                });
            }
        }

         public static void EnrichTrainingfile()
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv";
            string validfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\valid.csv";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features02\train_all_04.csv";
            var fid2usercnt = Loader.LoadFid2Usercnt();
            var url2freqcnt00 = Loader.LoadUrl2factcnt00();
            var url2freqcnt01 = Loader.LoadUrl2factcnt01();
            var url2freqcnt02 = Loader.LoadUrl2factcnt02();
            var url2freqcnt03 = Loader.LoadUrl2factcnt03();
            var word2doccnt = Loader.LoadWord2Doccnt();
            
            var fid2url = Loader.LoadFid2Url();
            var fid2words = Loader.LoadFid2Words();
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");
            var validuids = new HashSet<string>(Loader.LoadGroundTruth(validfile).Keys);
            var user2fact = Loader.LoadUserFacts(user2matches);

            List<string> uid_set = user2matches.Keys.ToList();

            HashSet<string> visited_uid = new HashSet<string>();

            Console.WriteLine("calculating time features...");
            Dictionary<string, TimeFeatureBag> user2timefeature = new Dictionary<string, TimeFeatureBag>();
            foreach (var uid in user2matches.Keys)
            {
                user2timefeature.Add(uid, GenTimeFeature(user2fact[uid]));
            }
            Console.WriteLine("calculating time features finished.");


            Console.WriteLine("calculating user data cnt...");
            Dictionary<string, Dictionary<string, int>> user2wordcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2fidcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2urlcnt = new Dictionary<string, Dictionary<string, int>>();
            foreach (var uid in uid_set)
            {
                Dictionary<string, int> fidcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> urlcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> wordcnt_usera = GetWordCnt(user2fact[uid], fid2words, fidcnt_usera, urlcnt_usera, fid2url, word2doccnt);
                user2wordcnt.Add(uid, wordcnt_usera);
                user2fidcnt.Add(uid, fidcnt_usera);
                user2urlcnt.Add(uid, urlcnt_usera);
            }
            Console.WriteLine("calculating user data cnt finished.");


            using (StreamReader rd = new StreamReader(infile))
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int cnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 1000 == 0)
                    {
                        Console.WriteLine("output pair {0}", cnt);
                    }
                    string[] words = content.Split(','); 

                    if (user2fact.ContainsKey(words[0]) && user2fact.ContainsKey(words[1]) && user2fact[words[0]].facts.Count > 5 && user2fact[words[1]].facts.Count > 5)
                    {
                        TimeFeatureBag tf_ua  = user2timefeature[words[0]];
                        TimeFeatureBag tf_ub = user2timefeature[words[1]];
                        WriteOneLine(words, fid2words, fid2url, user2fact, wt, word2doccnt, fid2usercnt, url2freqcnt00, url2freqcnt01, url2freqcnt02, url2freqcnt03, 1, tf_ua, tf_ub, user2wordcnt, user2fidcnt, user2urlcnt);

                        if (!visited_uid.Contains(words[0]))
                        {
                            visited_uid.Add(words[0]);
                            if (validuids.Contains(words[0]) || rng.NextDouble() < 0.00022)
                            {
                                // complete set
                                foreach (var neg_uid in uid_set)
                                {
                                    if (rng.NextDouble()>0.6 && user2fact.ContainsKey(neg_uid) && user2fact[neg_uid].facts.Count > 5 && neg_uid.CompareTo(words[0]) > 0 && !user2matches[words[0]].Contains(neg_uid))
                                    {
                                        tf_ub = user2timefeature[neg_uid];
                                        WriteOneLine(new string[] { words[0], neg_uid }, fid2words, fid2url, user2fact, wt, word2doccnt, fid2usercnt, url2freqcnt00, url2freqcnt01, url2freqcnt02, url2freqcnt03, 0, tf_ua, tf_ub, user2wordcnt, user2fidcnt, user2urlcnt);
                                    }
                                }
                            }
                            else
                            {
                                for (int k = 0; k < 10; k++)
                                {
                                    string neg_uid = SampleOneUid(words[0], user2matches, uid_set, user2fact);

                                    if (neg_uid == null)
                                        break;

                                    tf_ub = user2timefeature[neg_uid];
                                    WriteOneLine(new string[] { words[0], neg_uid }, fid2words, fid2url, user2fact, wt, word2doccnt, fid2usercnt, url2freqcnt00, url2freqcnt01, url2freqcnt02, url2freqcnt03, 0, tf_ua, tf_ub, user2wordcnt, user2fidcnt, user2urlcnt);
                                }
                            }
                        }

                        if (!visited_uid.Contains(words[1]))
                        {
                            visited_uid.Add(words[1]);
                            tf_ua = user2timefeature[words[1]];
                            for (int k = 0; k < 5; k++)
                            {
                                string neg_uid = SampleOneUid(words[1], user2matches, uid_set, user2fact);

                                if (neg_uid == null)
                                    break;

                                tf_ub = user2timefeature[neg_uid];
                                WriteOneLine(new string[] { words[1], neg_uid }, fid2words, fid2url, user2fact, wt, word2doccnt, fid2usercnt, url2freqcnt00, url2freqcnt01, url2freqcnt02, url2freqcnt03, 0, tf_ua, tf_ub, user2wordcnt, user2fidcnt, user2urlcnt);
                            }
                        }
                    }
                    else
                    {
                        Console.WriteLine("Missing user!"); 
                    }
                    
                }
            }
        }
        
        public static void GenTestFile()
        {
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features\test_enriched_labeled.csv";
            var fid2usercnt = Loader.LoadFid2Usercnt();
            var url2freqcnt00 = Loader.LoadUrl2factcnt00();
            var url2freqcnt01 = Loader.LoadUrl2factcnt01();
            var url2freqcnt02 = Loader.LoadUrl2factcnt02();
            var url2freqcnt03 = Loader.LoadUrl2factcnt03();
            var word2doccnt = Loader.LoadWord2Doccnt();
            var user2fact = Loader.LoadUserFacts();
            var fid2url = Loader.LoadFid2Url();
            var fid2words = Loader.LoadFid2Words();
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");

            List<string> uid_set = user2fact.Keys.ToList();
            int len = uid_set.Count;
            //using (StreamReader rd = new StreamReader(infile))
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int cnt = 0;
                for (int i = 0; i < len; i++)
                {
                    if (user2matches.ContainsKey(uid_set[i]) || user2fact[uid_set[i]].facts.Count <= 5)
                    {
                        continue;
                    }
                    if (cnt++ % 1000 == 0)
                    {
                        Console.WriteLine("output pair {0}", cnt);
                    }

                    List<Tuple<string, double, double, double, double, double, double>> curlist = new List<Tuple<string, double, double, double, double, double, double>>();

                    for (int j = 0; j < len; j++)
                    {
                        if (user2matches.ContainsKey(uid_set[j]) || uid_set[j].CompareTo(uid_set[i]) <= 0 || user2fact[uid_set[j]].facts.Count <= 5)
                        {
                            continue;
                        }

                        Dictionary<string, int> fidcnt_usera = new Dictionary<string, int>();
                        Dictionary<string, int> urlcnt_usera = new Dictionary<string, int>();
                        Dictionary<string, int> wordcnt_usera = GetWordCnt(user2fact[uid_set[i]], fid2words, fidcnt_usera, urlcnt_usera, fid2url);

                        if (user2fact.ContainsKey(uid_set[j]))
                        {

                            Dictionary<string, int> fidcnt_userb = new Dictionary<string, int>();
                            Dictionary<string, int> urlcnt_userb = new Dictionary<string, int>();
                            Dictionary<string, int> wordcnt_userb = GetWordCnt(user2fact[uid_set[j]], fid2words, fidcnt_userb, urlcnt_userb, fid2url);

                            double wordsim = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125);
                            double fidsim = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000);
                            double url00sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204);
                            double url01sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900);
                            double url02sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866);
                            double url03sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866);
                            if (wordsim >= 1e-4)
                                curlist.Add(new Tuple<string, double, double, double, double, double, double>(uid_set[j], wordsim, fidsim, url00sim, url01sim, url02sim, url03sim));
                        }
                    }
                    HashSet<string> visited = new HashSet<string>();
                    int topk = 100;
                    curlist.Sort((a, b) => b.Item2.CompareTo(a.Item2));
                    for (int k = 0; k < topk && k < curlist.Count; k++)
                    {
                        if (!visited.Contains(curlist[k].Item1))
                        {
                            visited.Add(curlist[k].Item1);
                            wt.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8}", 1, uid_set[i], curlist[k].Item1, curlist[k].Item2, curlist[k].Item3, curlist[k].Item4, curlist[k].Item5, curlist[k].Item6, curlist[k].Item7);
                        }
                    }
                    curlist.Sort((a, b) => b.Item3.CompareTo(a.Item3));
                    for (int k = 0; k < topk; k++)
                    {
                        if (!visited.Contains(curlist[k].Item1))
                        {
                            visited.Add(curlist[k].Item1);
                            wt.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8}", 1, uid_set[i], curlist[k].Item1, curlist[k].Item2, curlist[k].Item3, curlist[k].Item4, curlist[k].Item5, curlist[k].Item6, curlist[k].Item7);
                        }
                    }
                    curlist.Sort((a, b) => (b.Item4 + b.Item5 + b.Item6 + b.Item7).CompareTo(a.Item4 + a.Item5 + a.Item6 + a.Item7));
                    for (int k = 0; k < topk; k++)
                    {
                        if (!visited.Contains(curlist[k].Item1))
                        {
                            visited.Add(curlist[k].Item1);
                            wt.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8}", 1, uid_set[i], curlist[k].Item1, curlist[k].Item2, curlist[k].Item3, curlist[k].Item4, curlist[k].Item5, curlist[k].Item6, curlist[k].Item7);
                        }
                    }
                }
            }
        }
        */
        public static void EnrichTrainingfile(int fidx)
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_tree\train_candi_report" + fidx + ".csv";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\new_added_from_tree\train_tree_report_detailed_part" + fidx;
             
            var fid2usercnt = Loader.LoadFid2Usercnt();
            var url2freqcnt00 = Loader.LoadUrl2factcnt00();
            var url2freqcnt01 = Loader.LoadUrl2factcnt01();
            var url2freqcnt02 = Loader.LoadUrl2factcnt02();
            var url2freqcnt03 = Loader.LoadUrl2factcnt03();
            var word2doccnt = Loader.LoadWord2Doccnt();
            
            var fid2url = Loader.LoadFid2Url();
            var fid2words = Loader.LoadFid2Words();
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");            
            var user2fact = Loader.LoadUserFacts(user2matches);

            List<string> uid_set = user2matches.Keys.ToList();
             
            Console.WriteLine("calculating time features...");
            Dictionary<string, TimeFeatureBag> user2timefeature = new Dictionary<string, TimeFeatureBag>();
            foreach (var uid in uid_set)
            {
                user2timefeature.Add(uid, GenTimeFeature(user2fact[uid]));
            }
            Console.WriteLine("calculating time features finished.");


            Console.WriteLine("calculating user data cnt...");
            Dictionary<string, Dictionary<string, int>> user2wordcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2fidcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2urlcnt = new Dictionary<string, Dictionary<string, int>>();
            foreach (var uid in uid_set)
            {
                Dictionary<string, int> fidcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> urlcnt_usera = new Dictionary<string, int>();
                Dictionary<string, int> wordcnt_usera = GetWordCnt(user2fact[uid], fid2words, fidcnt_usera, urlcnt_usera, fid2url, word2doccnt);
                user2wordcnt.Add(uid, wordcnt_usera);
                user2fidcnt.Add(uid, fidcnt_usera);
                user2urlcnt.Add(uid, urlcnt_usera);
            }
            Console.WriteLine("calculating user data cnt finished.");


            var keyurls = UrlComAnalysis.LoadKeyUrls();
            int dep = keyurls.Count;
            var user2url = UrlComAnalysis.LoadUser2url(1, user2fact);
            string default_urlfeature = "";
            for (int i = 0; i < dep; i++)
            {
                default_urlfeature += ",0";
            }




            using (StreamReader rd = new StreamReader(infile))
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int cnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 1000 == 0)
                    {
                        Console.WriteLine("output pair {0}", cnt);
                    }
                    string[] words = content.Split(',');

                    if (user2fact.ContainsKey(words[1]) && user2fact.ContainsKey(words[2]) && user2fact[words[1]].facts.Count > 5 && user2fact[words[2]].facts.Count > 5)
                    {
                        TimeFeatureBag tf_ua = user2timefeature[words[1]];
                        TimeFeatureBag tf_ub = user2timefeature[words[2]];
                        WriteOneLine(new string[]{words[1],words[2]}, fid2words, fid2url, user2fact, wt, word2doccnt, fid2usercnt, url2freqcnt00, url2freqcnt01, url2freqcnt02, url2freqcnt03,
                            int.Parse(words[0]), tf_ua, tf_ub, user2wordcnt, user2fidcnt, user2urlcnt, user2url, default_urlfeature, dep, keyurls);
                    }
                    else
                    {
                        //Console.WriteLine("skipping pair.");
                    }              
                }
            }
        }

        public static void EnrichTestfile(int fidx)
        {
            //string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\split_new_added_test_inst\new_added_test_inst.csv.part" + fidx;
            //string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\keyurl_new_added_test_inst\test_part" + fidx;

            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features\test_enriched_labeled_LR\test_enriched_labeled_LR0.csv";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\consistent_test\test_part0";

            var fid2usercnt = Loader.LoadFid2Usercnt();
            var url2freqcnt00 = Loader.LoadUrl2factcnt00();
            var url2freqcnt01 = Loader.LoadUrl2factcnt01();
            var url2freqcnt02 = Loader.LoadUrl2factcnt02();
            var url2freqcnt03 = Loader.LoadUrl2factcnt03();
            var word2doccnt = Loader.LoadWord2Doccnt();

            var fid2url = Loader.LoadFid2Url();
            var fid2words = Loader.LoadFid2Words();
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");
            var user2fact = Loader.LoadUserFacts();

            List<string> uid_set = new List<string>(user2fact.Keys);
            
            Console.WriteLine("calculating time features...");
            Dictionary<string, TimeFeatureBag> user2timefeature = new Dictionary<string, TimeFeatureBag>();
            foreach (var uid in uid_set)
            {
                if (!user2matches.ContainsKey(uid))
                    user2timefeature.Add(uid, GenTimeFeature(user2fact[uid]));
            }
            Console.WriteLine("calculating time features finished.");


            Console.WriteLine("calculating user data cnt...");
            Dictionary<string, Dictionary<string, int>> user2wordcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2fidcnt = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, int>> user2urlcnt = new Dictionary<string, Dictionary<string, int>>();
            foreach (var uid in uid_set)
            {
                if (!user2matches.ContainsKey(uid))
                {
                    Dictionary<string, int> fidcnt_usera = new Dictionary<string, int>();
                    Dictionary<string, int> urlcnt_usera = new Dictionary<string, int>();
                    Dictionary<string, int> wordcnt_usera = GetWordCnt(user2fact[uid], fid2words, fidcnt_usera, urlcnt_usera, fid2url, word2doccnt);
                    user2wordcnt.Add(uid, wordcnt_usera);
                    user2fidcnt.Add(uid, fidcnt_usera);
                    user2urlcnt.Add(uid, urlcnt_usera);
                }
            }
            Console.WriteLine("calculating user data cnt finished.");

           
            var keyurls = UrlComAnalysis.LoadKeyUrls();
            int dep = keyurls.Count;
            var user2url = UrlComAnalysis.LoadUser2url(1, user2fact);
            string default_urlfeature = "";
            for (int i = 0; i < dep; i++)
            {
                default_urlfeature += ",0";
            }



            using (StreamReader rd = new StreamReader(infile))
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int cnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 1000 == 0)
                    {
                        Console.WriteLine("output pair {0}", cnt);
                    }
                    string[] words = content.Split(',');

                    string uida = words[0];
                    string uidb = words[1];

                    if (user2matches.ContainsKey(uida) || user2matches.ContainsKey(uidb))
                    {
                        continue;
                    }

                    if (user2fact.ContainsKey(uida) && user2fact.ContainsKey(uidb) && user2fact[uida].facts.Count > 5 && user2fact[uidb].facts.Count > 5)
                    { 
                        TimeFeatureBag tf_ua = user2timefeature[uida];
                        TimeFeatureBag tf_ub = user2timefeature[uidb];
                        WriteOneLine(new string[] { uida, uidb }, fid2words, fid2url, user2fact, wt, word2doccnt, fid2usercnt, url2freqcnt00, url2freqcnt01, url2freqcnt02, url2freqcnt03, 
                            0, tf_ua, tf_ub, user2wordcnt, user2fidcnt, user2urlcnt, user2url, default_urlfeature, dep, keyurls);
                    }
                    else
                    {
                        //Console.WriteLine("skipping pair.");
                    }
                }
            }
        }

        public static void ExtractUserFeatures( )
        {
            Dictionary<string, int> keyurl2groupidx = new Dictionary<string, int>();
            using (StreamReader rd = new StreamReader(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\common_user_stat\url_lift.csv"))
            {
                string content = null;
                int cnt = 0;
                while ((content = rd.ReadLine()) != null)
                {
                    
                    string[] words = content.Split(',');
                    keyurl2groupidx.Add(words[0], cnt / 100);
                    cnt++;
                    if (cnt >= 4000)
                    {
                        break;
                    }
                }
            }

            Dictionary<string, int> url2idx = new Dictionary<string, int>();
            using (StreamReader rd = new StreamReader(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\url_freq_dep00.csv"))
            {
                string content = null; 
                int cnt=0;
                while ((content = rd.ReadLine()) != null)
                { 
                    string[] words = content.Split(',');
                    url2idx.Add(words[0], cnt++);
                    if (cnt >= 500)
                    {
                        break;
                    }
                }
            }

            float[] mainurl2cnt = new float[500];
            float[] keyurlcnt = new float[40];
            float[] hour2cnt = new float[24];
            float[] day2cnt = new float[7];
            float[] month2cnt = new float[20];
           

            Dictionary<string, string> fact2url00 = Loader.LoadFid2Url(0);
            Dictionary<string, string> fact2url01 = Loader.LoadFid2Url(1);

            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\user_features.csv";

            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\facts.json";

            DateTime starttime = DateTime.Parse("2014-12-01");

            using(StreamWriter wt = new StreamWriter(outfile))
            using (StreamReader rd = new StreamReader(infile))
            { 
                int factcnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (factcnt++ % 10000 == 0)
                    {
                        Console.WriteLine(factcnt + "\tLoadUserFacts");
                    }

                    Facts ss = JsonConvert.DeserializeObject<Facts>(content);
                                        
                    ss.facts.Sort((a, b) => a.ts.CompareTo(b.ts));

                    for (int i = 0; i < 40; i++)
                    {
                        keyurlcnt[i] = 0;
                    }
                    for (int i = 0; i < 500; i++)
                    {
                        mainurl2cnt[i] = 0;
                    }
                    for (int i = 0; i < 24; i++)
                    {
                        hour2cnt[i] = 0;
                    }
                    for(int i=0;i<7;i++){
                        day2cnt[i]=0;
                    }
                    for (int i = 0; i < 20; i++)
                    {
                        month2cnt[i]=0;
                    }


                    float unit = 1.0f/ss.facts.Count;

                    foreach (var fact in ss.facts)
                    {
                        if (fact2url00.ContainsKey(fact.fid))
                        {
                            string url = fact2url00[fact.fid];
                            if (url2idx.ContainsKey(url))
                            {
                                mainurl2cnt[url2idx[url]] += unit;
                            }
                        }
                        if (fact2url01.ContainsKey(fact.fid))
                        {
                            string url = fact2url01[fact.fid];
                            if (keyurl2groupidx.ContainsKey(url))
                            {
                                keyurlcnt[keyurl2groupidx[url]] += unit;
                            }
                        }

                        DateTime ctime = UserProfileInfer.Utils.Common.ParseTimeStampMillisecond(fact.ts);
                        int houridx = ctime.Hour;
                        int day = (int)ctime.DayOfWeek;
                        int month = (int)(ctime.Subtract(starttime).TotalDays / 31);

                        hour2cnt[houridx] += unit;
                        day2cnt[day] += unit;
                        month2cnt[month] += unit;
                    }

                    wt.Write(ss.uid);
                    for(int i=0;i<40;i++){
                        wt.Write(",{0}",keyurlcnt[i]);
                    }
                    for(int i=0;i<500;i++){
                        wt.Write(",{0}", mainurl2cnt[i]);
                    }
                    for (int i = 0; i < 24; i++)
                    {
                        wt.Write(",{0}",hour2cnt[i]);
                    }
                    for (int i = 0; i < 7; i++)
                    {
                        wt.Write(",{0}", day2cnt[i]);
                    }
                    for (int i = 0; i < 20; i++)
                    {
                        wt.Write(",{0}", month2cnt[i]);
                    }
                    wt.WriteLine();
                }
            }

            
        }

        public static void GetValidfileCandida()
        {
            
            string validfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\valid.csv";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\candidate\valid_candi.csv";
            
             
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");
            var user2facts = Loader.LoadUserFacts(user2matches);
            var fid2url = Loader.LoadFid2Url(0);

            Dictionary<string, HashSet<string>> user2url = ExtractUser2Url(user2facts, fid2url );
      
            var validuids = new HashSet<string>();
            using (StreamReader rd = new StreamReader(validfile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    if (!validuids.Contains(words[0]))
                    {
                        validuids.Add(words[0]);
                    }
                }
            }
            List<string> all_uids = new List<string>(user2matches.Keys);

            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int cnt = 0;
                foreach (var uid in validuids)
                {
                    Console.WriteLine(cnt++ + "\t" + uid);

                    foreach (var muid in all_uids)
                    {
                        if (muid.CompareTo(uid) > 0)
                        {
                            bool falg = false;
                            if (user2url[uid].Count > 0)
                            {
                                foreach (var url in user2url[uid])
                                {
                                    if (user2url[muid].Contains(url))
                                    {
                                        falg = true;
                                        break;
                                    }
                                }
                            }
                            if(falg)
                                wt.WriteLine("{0},{1},{2}", user2matches[uid].Contains(muid) ? 1 : 0, uid, muid);
                        }
                    }
                }
            }
        }

        public static void GetTrainfileCandida()
        {

            string trainfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\candi\train_ranpair.csv";


            var user2matches = Loader.LoadGroundTruth(trainfile);

             

            HashSet<string> uid_set = new HashSet<string>();
            foreach (var pair in user2matches)
            {
                foreach (var mid in pair.Value)
                {
                    if (pair.Key.CompareTo(mid) < 0)
                    {
                        uid_set.Add(pair.Key);
                        break;
                    }
                }
            }

            List<string> all_uids = new List<string>(user2matches.Keys);
            int len = all_uids.Count;
            Random rng = new Random((int)DateTime.Now.Ticks);
            int k = 50;

            int cnt = 0;
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                foreach (var uid in uid_set)
                {
                    if (cnt++ % 10000 == 0)
                    {
                        Console.WriteLine(cnt);
                    }
                    for (int i = 0; i < k; i++)
                    {
                        int idx = rng.Next(len);
                        if (all_uids[idx].CompareTo(uid) > 0)
                        {
                            wt.WriteLine("{0},{1},{2}", user2matches[uid].Contains(all_uids[idx]) ? 1 : 0, uid, all_uids[idx]);
                        }
                    }
                }
            }


        }

        private static Dictionary<string, HashSet<string>> ExtractUser2Url(Dictionary<string, Facts> user2facts, Dictionary<string, string> fid2url)
        {
            Dictionary<string, HashSet<string>> user2urls = new Dictionary<string, HashSet<string>>();
            foreach (var pair in user2facts)
            {
                HashSet<string> curl_set = new HashSet<string>();
                foreach (var fact in pair.Value.facts)
                {
                    if (fid2url.ContainsKey(fact.fid))
                    {
                        if (!curl_set.Contains(fid2url[fact.fid]))
                        {
                            curl_set.Add(fid2url[fact.fid]);
                        }
                    }
                }
                user2urls.Add(pair.Key, curl_set);
            }
            return user2urls;
        }
 


        public static void WriteOneLine(string[] words, Dictionary<string, List<string>> fid2words, Dictionary<string, List<string>> fid2url, 
            Dictionary<string, Facts> user2fact, StreamWriter wt, Dictionary<string, int> word2doccnt,
            Dictionary<string, int> fid2usercnt, Dictionary<string, int> url2freqcnt00, Dictionary<string, int> url2freqcnt01,
            Dictionary<string, int> url2freqcnt02, Dictionary<string, int> url2freqcnt03, int label,
            TimeFeatureBag tf_ua, TimeFeatureBag tf_ub,
            Dictionary<string, Dictionary<string, int>> user2wordcnt, Dictionary<string, Dictionary<string, int>> user2fidcnt, Dictionary<string, Dictionary<string, int>> user2urlcnt
            , Dictionary<string, HashSet<string>> user2urls, string defaulturlffeature, int urldep, List<Dictionary<string,double>> keyurls
            )
        {
            Dictionary<string, int> fidcnt_usera = user2fidcnt[words[0]];
            Dictionary<string, int> urlcnt_usera = user2urlcnt[words[0]];
            Dictionary<string, int> wordcnt_usera = user2wordcnt[words[0]];

            Dictionary<string, int> fidcnt_userb = user2fidcnt[words[1]];
            Dictionary<string, int> urlcnt_userb = user2urlcnt[words[1]];
            Dictionary<string, int> wordcnt_userb = user2wordcnt[words[1]];

            double wordsim = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125);
            double fidsim = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000);
            double url00sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204);
            double url01sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900);
            double url02sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866);
            double url03sim = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866);

            string timefeatureline = GenTimeFeatureLine(words[0], words[1], user2fact, tf_ua, tf_ub);

            //--
            double wordsim02 = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125,2);
            double fidsim02 = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000, 2);
            double url00sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204, 2);
            double url01sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900, 2);
            double url02sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866, 2);
            double url03sim02 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866, 2);

            double wordsim03 = CalcTFIDF(wordcnt_usera, wordcnt_userb, word2doccnt, 300125, 10, 50);
            double fidsim03 = CalcTFIDF(fidcnt_usera, fidcnt_userb, fid2usercnt, 300000, 10, 50);
            double url00sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt00, 4066204, 10, 50);
            double url01sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt01, 701900, 10, 50);
            double url02sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt02, 154866, 10, 50);
            double url03sim03 = CalcTFIDF(urlcnt_usera, urlcnt_userb, url2freqcnt03, 104866, 10, 50);

            int comFidCnt = 0;
            int fidSum = fidcnt_usera.Count + fidcnt_userb.Count;
            foreach (var pair in fidcnt_usera)
            {
                if (pair.Value > 1 && fidcnt_userb.ContainsKey(pair.Key))
                {
                    comFidCnt++;
                }
            }
            double comFidRatio = comFidCnt*1.0 / fidSum;

            int comUrlCnt = 0;
            int urlSum = urlcnt_usera.Count + urlcnt_userb.Count;
            foreach (var pair in urlcnt_usera)
            {
                if (pair.Value > 1 && urlcnt_userb.ContainsKey(pair.Key))
                {
                    comUrlCnt++;
                }
            }
            double comUrlRatio = comUrlCnt*1.0 / urlSum;

            wt.Write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26}",
                label, words[0], words[1], wordsim, fidsim, url00sim, url01sim, url02sim, url03sim, timefeatureline  
                ,Math.Min(user2fact[words[0]].facts.Count, user2fact[words[1]].facts.Count)
                , wordsim02, fidsim02, url00sim02, url01sim02, url02sim02, url03sim02
                , wordsim03, fidsim03, url00sim03, url01sim03, url02sim03, url03sim03
                , comFidCnt, comFidRatio, comUrlCnt, comUrlRatio
                );

            if (user2urls.ContainsKey(words[0]) && user2urls.ContainsKey(words[1]))
            {
                for (int i = 0; i < urldep; i++)
                {
                    int cnt = 0;
                    foreach (var keyurl in keyurls[i].Keys)
                    {
                        if (user2urls[words[0]].Contains(keyurl) && user2urls[words[1]].Contains(keyurl))
                        {
                            cnt++;
                        }
                    }
                    wt.Write("," + cnt);
                }
                wt.WriteLine();
            }
            else
            {
                wt.WriteLine(defaulturlffeature);
            }

        }




        public static string SampleOneUid(string uid, Dictionary<string, HashSet<string>> user2matches, List<string> uid_set, Dictionary<string, Facts> user2fact)
        {
            int len = uid_set.Count;
            int failurecnt = 0;
            while (true)
            {
                int idx = rng.Next(len);
                if (uid_set[idx].CompareTo(uid) > 0 && !user2matches[uid].Contains(uid_set[idx]) && user2fact.ContainsKey(uid_set[idx]) && user2fact[uid_set[idx]].facts.Count > 5)
                {
                    return uid_set[idx];
                }
                failurecnt++;
                if (failurecnt > 10)
                {
                    break;
                }
            }
            return null;
        }

        public static double CalcTFIDF(Dictionary<string, int> dicta, Dictionary<string, int> dictb, Dictionary<string, int> word2doccnt, int N, int log_base=10, int denoise = 0)
        {
            double score = 0;
            double suma = dicta.Sum(a => a.Value) + denoise;
            double sumb = dictb.Sum(a => a.Value) + denoise;
            foreach (var word in dicta.Keys)
            {
                if (word2doccnt.ContainsKey(word) && dictb.ContainsKey(word))
                {
                    score += Math.Sqrt(dicta[word] / suma * dictb[word] / sumb) * Math.Log(  N * 1.0 / word2doccnt[word], log_base);
                }
            }
            return score;
        }

        public static Dictionary<string, int> GetWordCnt(Facts fact, Dictionary<string, List<string>> fid2words, 
            Dictionary<string, int> fidcnt_user, Dictionary<string, int> urlcnt_user, 
            Dictionary<string, List<string>> fid2url, Dictionary<string,int> word2doccnt)
        {
            Dictionary<string, int> res = new Dictionary<string, int>();
            foreach (var ss in fact.facts)
            {

                if (!fidcnt_user.ContainsKey(ss.fid))
                {
                    fidcnt_user.Add(ss.fid, 1);
                }
                else
                {
                    fidcnt_user[ss.fid]++;
                }

                if (fid2url.ContainsKey(ss.fid))
                {
                    foreach (var curl in fid2url[ss.fid])
                    {
                        if (!urlcnt_user.ContainsKey(curl))
                        {
                            urlcnt_user.Add(curl, 1);
                        }
                        else
                        {
                            urlcnt_user[curl]++;
                        }
                    }
                }

                var words = fid2words.ContainsKey(ss.fid) ? fid2words[ss.fid] : null;
                if (words != null)
                {
                    foreach (var word in words)
                    {
                        if (word2doccnt.ContainsKey(word) && word2doccnt[word] < 100000)
                        {
                            if (!res.ContainsKey(word))
                            {
                                res.Add(word, 1);
                            }
                            else
                            {
                                res[word]++;
                            }
                        }
                    }
                }
            }
            return res;
        }

        #region pred models
        public static double FT100Pred(double f0, double f1, double f2, double f3, double f4, double f5, double f6, double f7, double f8, double f9, double f10
           , double f11, double f12, double f13, double f14, double f15, double f16, double f17, double f18, double f19, double f20, double f21
            , double f22, double f23, double f24, double f25, double f26, double f27, double f28, double f29, double f30, double f31, double f32)
        {
            double treeOutput0 = ((f32 > 0.05562594) ? ((f1 > 1.237585) ? ((f31 > 8.5) ? ((f12 > 0.7677026) ? ((f6 > -0.05401162) ? 0.785663420360408 : 0.332272026267412) : -0.247140918782666) : ((f1 > 1.584679) ? ((f27 > 0.7585866) ? -0.378463696948941 : 0.489725318589296) : ((f19 > 2.898992) ? 0.0727716377206288 : ((f25 > 0.3381263) ? -0.566192063011404 : 0.655473472128905)))) : ((f6 > 0.2736513) ? ((f31 > 11.5) ? ((f14 > 0.2837022) ? ((f7 > 0.1055546) ? -0.542379381759731 : 0.00744740271830372) : ((f18 > 2.167583) ? 0.553374642870307 : 0.125490254559641)) : ((f14 > 0.2240143) ? -0.694141599898826 : ((f2 > 0.725315) ? ((f29 > 2.5) ? 0.219847105224858 : -0.193368948614997) : -0.400979238402731))) : ((f32 > 0.1121388) ? ((f14 > 0.1344633) ? -0.747964526127036 : ((f5 > 0.2696638) ? ((f30 > 0.04128736) ? -0.431357340720047 : 0.580502931343692) : ((f31 > 12.5) ? ((f29 > 2.5) ? 0.0967602708316981 : -0.392029761569288) : ((f2 > 0.6696985) ? ((f29 > 1.5) ? -0.246828636758034 : -0.605626538506379) : -0.650306897305089)))) : ((f6 > -0.06560265) ? ((f31 > 10.5) ? ((f18 > 2.726756) ? ((f25 > 0.347221) ? -0.111799350121874 : 0.876475930971847) : ((f14 > 0.2784238) ? -0.744952287869009 : -0.36037418871411)) : -0.666947942567178) : -0.796377318226813)))) : ((f32 > 0.0294245) ? ((f24 > 0.7636249) ? ((f25 > 0.3606755) ? ((f6 > 0.3669006) ? 0.0860145303585624 : -0.576102493515047) : 0.627983153954179) : ((f6 > 0.331602) ? ((f31 > 9.5) ? ((f18 > 2.040148) ? 0.080586612458986 : -0.46517882689586) : -0.687051536276117) : ((f6 > -0.09045272) ? -0.796440162997916 : -0.921206590361479))) : ((f1 > 0.01911136) ? -0.90955177222885 : -0.98217801672191)));
            double treeOutput1 = ((f32 > 0.05265083) ? ((f18 > 3.998626) ? ((f31 > 8.5) ? ((f14 > 0.6630037) ? -0.214141742232416 : ((f6 > 0.1091064) ? 0.690712969070267 : 0.33961940411502)) : ((f18 > 4.979078) ? ((f12 > 0.5768576) ? ((f27 > 0.7585866) ? -0.324088435730282 : 0.426057329819166) : -0.509078688969102) : -0.233428255121241)) : ((f6 > 0.3669006) ? ((f31 > 9.5) ? ((f14 > 0.3875782) ? ((f7 > 0.1890608) ? -0.581247636813049 : -0.0364637399148497) : ((f18 > 1.946636) ? 0.409678313578845 : 0.0428474354386936)) : ((f25 > 0.5855793) ? -0.101354645952381 : -0.417937725420801)) : ((f6 > -0.09045272) ? ((f31 > 7.5) ? ((f14 > 0.1983607) ? ((f7 > 0.0286411) ? -0.672391541254366 : -0.374136784904661) : ((f19 > 2.326932) ? ((f10 > 17.82647) ? 0.187151636981624 : -0.181186152538522) : ((f31 > 15.5) ? ((f30 > 0.01667181) ? 0.198395367447419 : -0.374732976982489) : ((f25 > 0.2489905) ? -0.419929433481914 : ((f18 > 1.880638) ? 0.602704549522195 : -0.328982314241772))))) : ((f19 > 2.165528) ? ((f10 > 17.49965) ? -0.332629848366099 : -0.590464127537051) : -0.642843383889797)) : ((f10 > 17.7122) ? ((f31 > 7.5) ? -0.401040098486953 : -0.651120711711311) : -0.708049319798469)))) : ((f18 > 2.85683) ? ((f31 > 10.5) ? ((f25 > 0.378284) ? -0.0334556549661997 : 0.703674831088845) : ((f25 > 0.3056822) ? ((f1 > 1.82885) ? 0.290670504971647 : -0.616380580102315) : ((f29 > 2.5) ? 0.298981401714235 : -0.384836539425348))) : ((f32 > 0.02486295) ? ((f6 > 0.4329614) ? ((f31 > 10.5) ? -0.256458576727065 : -0.566445034230092) : ((f6 > -0.1030164) ? ((f23 > 1.952857) ? 0.636391505996204 : -0.683355932377494) : -0.783464351241018)) : -0.808562337735878)));
            double treeOutput2 = ((f32 > 0.06251879) ? ((f18 > 4.381141) ? ((f12 > 0.6421817) ? ((f31 > 6.5) ? ((f28 > 0.6536086) ? -0.338326879848594 : 0.543724626505583) : 0.0537267852399341) : -0.324962027405422) : ((f9 > 7.576229) ? ((f10 > 17.64495) ? ((f31 > 7.5) ? ((f25 > 0.5807843) ? ((f7 > -0.2119981) ? 0.201474859085964 : -0.137517697823508) : -0.285718465018099) : ((f2 > 0.6696985) ? -0.31080717973371 : -0.531846013132963)) : ((f14 > 0.2301387) ? -0.625970093146201 : ((f7 > 0.965847) ? -0.184668722596205 : ((f29 > 2.5) ? ((f2 > 0.265358) ? -0.408058564774249 : 0.117989803423313) : -0.566629884139977)))) : ((f6 > 0.4873235) ? ((f12 > 0.8754225) ? ((f31 > 13.5) ? 0.45216252388308 : ((f2 > 0.6577429) ? 0.23771838124789 : -0.10880600989084)) : -0.521251539126589) : ((f31 > 13.5) ? ((f14 > 0.4027027) ? -0.351115707461101 : ((f1 > 0.7968202) ? 0.389126261258702 : -0.000939387612360399)) : ((f10 > 3.467698) ? -0.176708856411101 : -0.451937193691204))))) : ((f18 > 2.370383) ? ((f31 > 12.5) ? ((f25 > 0.3426999) ? ((f6 > 0.1586699) ? 0.273086600527552 : -0.395076985404317) : 0.658971217895707) : ((f18 > 4.550251) ? ((f25 > 0.4490376) ? -0.155970498781786 : 0.618180288047108) : ((f6 > 0.5739934) ? -0.0405979724096385 : ((f25 > 0.2760238) ? -0.571050733735343 : ((f25 > 0.05093353) ? 0.250193255864382 : -0.414252603622567))))) : ((f25 > 0.05093353) ? ((f6 > 0.5323637) ? ((f31 > 8.5) ? -0.170473242855388 : -0.470176353343567) : ((f31 > 10.5) ? -0.492243830656562 : ((f10 > 17.77325) ? ((f7 > -0.1876039) ? ((f24 > 0.01471547) ? -0.35454012328125 : -0.618554460617831) : -0.633480275523557) : -0.662503146031142))) : -0.720727441460788)));
            double treeOutput3 = ((f30 > 0.02128645) ? ((f18 > 3.718299) ? ((f6 > 0.02181277) ? ((f31 > 6.5) ? ((f14 > 0.5988506) ? ((f7 > 0.7322611) ? -0.570014437693019 : 0.293247023705093) : ((f25 > 0.4038677) ? 0.442683073674806 : 0.741939172278568)) : ((f1 > 1.691513) ? 0.417842810330679 : -0.12202754512907)) : ((f19 > 1.388229) ? ((f3 > 0.5493562) ? ((f10 > 17.34971) ? 0.211645979231658 : -0.199507410060191) : -0.39259795293027) : 0.601250594113653)) : ((f6 > 0.1788875) ? ((f31 > 8.5) ? ((f14 > 0.2837022) ? ((f19 > 1.091446) ? ((f7 > 0.1055546) ? -0.515467438488034 : -0.113684429138898) : ((f18 > 1.755472) ? 0.794313919636176 : -0.265491365639404)) : ((f31 > 15.5) ? 0.442826101278684 : ((f18 > 1.92418) ? ((f28 > 0.1045845) ? ((f25 > 0.2599556) ? -0.240965505628663 : 0.712213871998905) : 0.405186652276681) : -0.0427457781982242))) : ((f14 > 0.1771242) ? -0.517241154853677 : ((f18 > 1.902291) ? -0.0450993806624631 : -0.30955456439693))) : ((f32 > 0.09575653) ? ((f14 > 0.08421986) ? -0.480432302262338 : ((f6 > -0.06400484) ? -0.0190616002353283 : -0.265375505356781)) : ((f31 > 9.5) ? ((f25 > 0.2260532) ? -0.390795632822152 : ((f18 > 1.79629) ? 0.586731022876548 : -0.249483654323597)) : -0.504986584549687)))) : ((f32 > 0.03587122) ? ((f5 > 0.2652187) ? ((f32 > 0.1430221) ? ((f1 > 0.6366574) ? -0.301825124455911 : 0.644956382790488) : -0.202468729926168) : ((f9 > 6.171175) ? ((f6 > -0.1086949) ? -0.457826258794715 : -0.600444289116544) : ((f6 > 0.5739934) ? ((f31 > 15.5) ? 0.292901418130964 : -0.148484238159823) : ((f31 > 15.5) ? -0.152759693454039 : -0.40596909831022)))) : ((f25 > 0.04420508) ? -0.583433313902893 : -0.661987045196847)));
            double treeOutput4 = ((f32 > 0.0468849) ? ((f1 > 0.9880317) ? ((f2 > 0.3908285) ? ((f6 > 0.221998) ? ((f2 > 0.8442891) ? ((f22 > 0.7094427) ? 0.0296783024245316 : 0.449091947237941) : ((f31 > 11.5) ? 0.368060875687377 : -0.189613124722607)) : ((f28 > 0.1562543) ? ((f31 > 17.5) ? 0.193255032906145 : -0.464334236110254) : ((f31 > 6.5) ? ((f30 > 0.02577597) ? ((f4 > 0.5207987) ? ((f0 > 1.27053) ? 0.208253686448844 : -0.549868565578799) : 0.343167624525906) : -0.274908870773849) : ((f19 > 2.898992) ? -0.0363074307577743 : -0.445873562317897)))) : 0.657794824257663) : ((f9 > 7.925733) ? ((f32 > 0.1430221) ? ((f14 > 0.05195682) ? -0.410004061302724 : ((f5 > 0.2533208) ? ((f0 > 0.8024864) ? -0.0356467589289047 : 0.606064712390933) : ((f29 > 1.5) ? -0.0491037058040198 : -0.324599175910787))) : ((f14 > 0.1206061) ? -0.518227015557522 : ((f29 > 2.5) ? ((f6 > -0.07483575) ? -0.133981151126431 : -0.344159786309502) : -0.422032478141873))) : ((f14 > 0.3310502) ? ((f10 > 2.42037) ? -0.185652867707089 : -0.493749571732138) : ((f29 > 2.5) ? ((f15 > 0.449443) ? ((f31 > 14.5) ? 0.382000070731009 : ((f28 > 0.06334205) ? -0.13991271243141 : ((f18 > 1.676616) ? 0.347540985411246 : -0.0152940963400065))) : ((f31 > 21.5) ? 0.254370855233529 : -0.188168661110118)) : ((f6 > 0.6505057) ? 0.0507188119844653 : ((f31 > 5.5) ? -0.201672833104056 : -0.416777498611412)))))) : ((f32 > 0.02027228) ? ((f6 > 0.2448097) ? ((f1 > 1.00746) ? ((f29 > 2.5) ? 0.327891316150031 : -0.202388957009723) : ((f14 > 0.3036891) ? -0.524537916306995 : ((f31 > 15.5) ? 0.106303421808947 : -0.358324425486047))) : -0.515989115031236) : -0.607392872332117));
            double treeOutput5 = ((f1 > 0.6860125) ? ((f31 > 8.5) ? ((f25 > 0.3289494) ? ((f22 > 0.4177666) ? ((f1 > 1.369758) ? 0.22059581989462 : ((f31 > 18.5) ? 0.0565447718737691 : ((f10 > 18.4178) ? ((f25 > 0.6607108) ? 0.31011735527144 : -0.254577798641433) : -0.444270504865005))) : ((f29 > 2.5) ? ((f14 > 0.2656863) ? ((f6 > 0.4329614) ? 0.364866618428099 : -0.104651902401337) : ((f4 > 0.4441568) ? ((f0 > 1.197571) ? 0.377964230098505 : -0.348437199636329) : 0.425276914685137)) : -0.177442081522086)) : ((f30 > 0.01074824) ? 0.622662343566004 : -0.277092862430368)) : ((f10 > 17.88258) ? ((f7 > -0.1876039) ? ((f25 > 0.7179731) ? 0.353736602924298 : ((f28 > 0.1181903) ? -0.310565159701503 : 0.122822991733759)) : -0.190268942101009) : ((f1 > 1.691513) ? ((f28 > 0.6536086) ? -0.666122757082149 : 0.248408025667289) : ((f25 > 0.2545401) ? ((f14 > 0.2240143) ? -0.52089138520873 : ((f7 > 0.9779225) ? 0.123277597052643 : -0.346591212905566)) : ((f30 > 0.02648656) ? 0.241312986896797 : -0.282699585736997))))) : ((f32 > 0.1000589) ? ((f22 > 0.9088575) ? ((f30 > 0.03816027) ? -0.267055587023821 : 0.480312191410169) : ((f6 > -0.06719974) ? ((f31 > 5.5) ? ((f14 > 0.1550481) ? -0.263682737289118 : 0.0484330213342713) : -0.27363868241624) : -0.303057900168816)) : ((f1 > 0.01911136) ? ((f10 > 17.7122) ? ((f7 > -0.1946847) ? ((f6 > -0.07182601) ? ((f23 > 1.747192) ? 1.26196834350311 : -0.0209789419742442) : -0.30253464970495) : -0.3879571383447) : ((f31 > 13.5) ? ((f14 > 0.3009434) ? -0.429589439562639 : ((f6 > 0.1054733) ? -0.00328197021913489 : -0.321492417475272)) : ((f6 > 0.7000832) ? -0.165398932812179 : -0.468757708671193))) : -0.574868932010494)));
            double treeOutput6 = ((f31 > 7.5) ? ((f1 > 0.8882377) ? ((f6 > -0.01908224) ? ((f14 > 0.5270468) ? ((f31 > 17.5) ? 0.299976997384442 : -0.31863955917126) : ((f28 > 0.1510706) ? ((f25 > 0.365082) ? ((f31 > 17.5) ? 0.388116138394903 : ((f1 > 1.369758) ? 0.189608950421871 : -0.241225837411881)) : 0.570361770064745) : ((f30 > 0.01550034) ? ((f30 > 0.1378544) ? -0.0166904884035353 : 0.458680919791792) : -0.0238861010848078))) : ((f25 > 0.3562232) ? ((f28 > 0.1277409) ? -0.337720789865425 : ((f30 > 0.02720838) ? 0.121344789031288 : -0.365426773413194)) : 0.443783890048717)) : ((f10 > 17.49965) ? ((f7 > -0.2060614) ? ((f6 > -0.06400484) ? ((f25 > 0.4862599) ? 0.316301593197353 : 0.0802949995429991) : -0.0556149973072013) : -0.151816315966774) : ((f6 > 0.4630982) ? ((f14 > 0.2148352) ? -0.196124370422269 : ((f18 > 1.302372) ? 0.226884806809644 : -0.05421835614653)) : ((f14 > 0.2519841) ? -0.41368701388886 : ((f6 > 0.02800021) ? ((f30 > 0.01315249) ? ((f31 > 20.5) ? 0.327159922900738 : -0.0738944663891055) : -0.289035048789522) : ((f7 > 0.9878271) ? 0.100220374214072 : -0.363879656671038)))))) : ((f19 > 2.47036) ? ((f6 > 0.5525283) ? ((f2 > 1.069078) ? 0.362901253159023 : 0.0331470782497078) : ((f10 > 17.42497) ? ((f7 > -0.1740668) ? ((f25 > 1.092159) ? 0.531431957272971 : ((f31 > 4.5) ? ((f0 > 1.696755) ? 0.588893176568665 : 0.072789778267451) : -0.181040793470676)) : -0.223383537093252) : ((f7 > 0.9981551) ? -0.0126788315456371 : -0.366061757265957))) : ((f18 > 6.075307) ? 0.505666834695613 : ((f31 > 5.5) ? ((f10 > 18.4178) ? ((f7 > -0.1680053) ? -0.00935864819986431 : -0.264867629984249) : -0.399997562684538) : -0.509137946443268))));
            double treeOutput7 = ((f30 > 0.01961497) ? ((f31 > 14.5) ? ((f23 > 0.2536597) ? ((f25 > 0.2812213) ? ((f6 > 0.05699217) ? ((f4 > 1.519449E-05) ? ((f28 > 0.1291643) ? ((f1 > 1.071631) ? 0.367447742393249 : -0.0585495398254791) : ((f1 > 0.540709) ? 0.443539192003272 : 0.140557045192793)) : -0.408763906706844) : ((f10 > 17.99229) ? 0.280861547064607 : -0.101869295108891)) : 0.604821634172839) : -0.392517680896739) : ((f6 > 0.1352799) ? ((f14 > 0.4132458) ? ((f7 > 0.2519511) ? -0.500994822626396 : -0.121489413690847) : ((f1 > 0.7744365) ? ((f15 > 0.4029925) ? ((f28 > 0.1381071) ? ((f25 > 0.2708267) ? ((f3 > 0.4464557) ? 0.159332103253615 : -0.279829817373489) : 0.557953595136571) : ((f29 > 1.5) ? 0.347149517995793 : -0.220157574950359)) : ((f25 > 0.3056822) ? -0.125387284453995 : 0.465242392522751)) : ((f6 > 0.7000832) ? 0.169683314687012 : ((f31 > 5.5) ? ((f27 > 0.06083602) ? -0.11464473312313 : 0.120745712608387) : -0.236993114154782)))) : ((f6 > -0.1280944) ? ((f14 > 0.1606452) ? ((f7 > 0.02132078) ? -0.438723675128923 : -0.213042752955852) : ((f1 > 1.428488) ? 0.239163916355818 : ((f2 > 0.7004806) ? ((f31 > 5.5) ? 0.0376491298445764 : -0.191865000396715) : ((f30 > 0.08476043) ? -0.438942110438633 : -0.192005574011233)))) : -0.381342943659387))) : ((f32 > 0.03347) ? ((f6 > -0.08207309) ? ((f5 > 0.2570927) ? 0.20632567728988 : ((f6 > 0.6659548) ? 0.0222733552818619 : ((f1 > 0.01355108) ? ((f14 > 0.3623737) ? -0.380636367527143 : ((f31 > 18.5) ? 0.0622019080462776 : ((f23 > 0.4305942) ? ((f0 > 2.36744) ? 0.609767997550306 : -0.250247995442273) : -0.0914890200811493))) : -0.38979817691698))) : -0.406772872733183) : -0.491218350285629));
            double treeOutput8 = ((f32 > 0.07273777) ? ((f9 > 9.908737) ? ((f10 > 18.4178) ? ((f31 > 12.5) ? 0.168602231131568 : ((f7 > -0.1666668) ? 0.0156762797298746 : -0.168264954347315)) : ((f2 > 0.2788037) ? ((f28 > 0.08055306) ? -0.330771410974341 : ((f29 > 3.5) ? -0.0443111211577237 : -0.250377535225268)) : ((f24 > 0.3102335) ? 0.49879117955832 : -0.168486699351615))) : ((f10 > 3.130651) ? ((f31 > 10.5) ? ((f3 > 0.4180971) ? 0.34071876212103 : ((f25 > 0.2599556) ? 0.0948696202864783 : 0.568597528476109)) : ((f19 > 2.146651) ? ((f29 > 3.5) ? 0.286871113841099 : 0.0701329717465366) : ((f25 > 0.2082298) ? -0.132502694345625 : 0.23093050815373))) : ((f12 > 4.920364) ? ((f7 > 0.923084) ? 0.321440842819575 : ((f31 > 20.5) ? 0.253026305121984 : -0.101588176156073)) : -0.317925918279814))) : ((f1 > 1.071631) ? ((f25 > 0.4080476) ? ((f14 > 0.2956891) ? -0.347278755846737 : ((f6 > 0.4329614) ? 0.289942701740662 : -0.0648702334380282)) : 0.422239079746682) : ((f25 > 0.03853185) ? ((f6 > 0.2175762) ? ((f31 > 10.5) ? ((f14 > 0.43534) ? -0.287617908335994 : ((f31 > 24.5) ? 0.372208290502195 : ((f16 > 210.5) ? -0.240762104649913 : ((f15 > 0.3149198) ? 0.229560006400061 : -0.0208239921348248)))) : ((f6 > 0.8226977) ? 0.15788047939998 : ((f14 > 0.2615823) ? -0.380159846651643 : -0.185349018170217))) : ((f6 > -0.1174213) ? ((f23 > 1.952857) ? ((f0 > 4.415589) ? 1.12549524361436 : -0.176506095768763) : ((f14 > 0.3036891) ? -0.418310277516313 : ((f31 > 11.5) ? ((f25 > 0.2489905) ? -0.15584755498696 : ((f1 > 0.4327258) ? 0.713789241491106 : 0.0295693929539637)) : -0.302577790353006))) : -0.461353367351943)) : -0.502602092151725)));
            double treeOutput9 = ((f29 > 2.5) ? ((f1 > 1.369758) ? ((f27 > 0.7585866) ? -0.195713656541119 : ((f13 > 0.6445313) ? 0.331352219122407 : ((f26 > 0.4924484) ? 0.331586761561299 : -0.319230002298892))) : ((f10 > 16.84369) ? ((f9 > 12.65169) ? ((f7 > -0.2021889) ? ((f30 > 0.09474178) ? -0.290002403821753 : ((f31 > 14.5) ? 0.394467882841684 : 0.0305090221039127)) : -0.197760710762645) : ((f7 > -0.1984221) ? ((f19 > 1.882107) ? 0.376758476981843 : ((f28 > 0.1034334) ? -0.12527187004908 : 0.289869024018947)) : 0.0305426990599057)) : ((f31 > 17.5) ? ((f14 > 0.43534) ? -0.175460145275545 : ((f18 > 1.097) ? ((f25 > 0.3996871) ? ((f30 > 0.07896698) ? -0.3424956927223 : ((f1 > 0.808596) ? 0.370443882618635 : 0.0498787302771619)) : 0.513537276983593) : -0.0943692803571245)) : ((f14 > 0.2063895) ? ((f25 > 0.3381263) ? -0.352733731663694 : ((f18 > 1.946636) ? 0.258302565856678 : -0.323234228376449)) : ((f7 > 0.9878271) ? 0.261017945741619 : ((f15 > 0.47391) ? ((f16 > 114.5) ? -0.186907860568825 : ((f9 > 11.78133) ? -0.111025360271008 : 0.161150426023229)) : -0.227719630876665)))))) : ((f32 > 0.1668385) ? ((f5 > 0.2363022) ? ((f23 > 0.6682917) ? -0.120993450375503 : 0.489107075958014) : ((f25 > 1.258936) ? 0.701967263860529 : -0.0703571274611917)) : ((f1 > 0.01609565) ? ((f10 > 17.10492) ? ((f7 > -0.1844412) ? ((f6 > -0.07483575) ? ((f25 > 0.8051723) ? 0.358717212807882 : 0.0422602270062212) : -0.157873485743237) : ((f23 > 0.2438893) ? -0.276870056157654 : -0.071614297130438)) : ((f7 > 0.9878271) ? ((f12 > 5.978814) ? 0.118306779216254 : -0.387468189147685) : ((f23 > 0.3932374) ? -0.35429082082854 : -0.260533687913715))) : -0.478996025208441)));
            double treeOutput10 = ((f31 > 6.5) ? ((f1 > 0.7968202) ? ((f28 > 0.1412316) ? ((f25 > 0.3426999) ? ((f31 > 23.5) ? 0.361196658503243 : ((f1 > 1.49885) ? ((f26 > 1.080976) ? -0.597431925692339 : ((f12 > 0.5137674) ? 0.282071167795059 : -0.360237062380431)) : -0.256113190086146)) : 0.444333303054231) : ((f25 > 0.4732654) ? ((f29 > 1.5) ? ((f15 > 0.5019789) ? ((f21 > 1.650523) ? ((f0 > 1.27053) ? 0.294337379154216 : -0.330834477023341) : ((f14 > 0.5031645) ? 0.00873082226638066 : 0.336151288232177)) : ((f6 > 0.470979) ? 0.302070735557828 : -0.0128637690552153)) : -0.293843012411734) : 0.510653471936159)) : ((f6 > 0.5972186) ? ((f2 > 0.6819611) ? 0.278627926761829 : 0.0621184934588743) : ((f32 > 0.1363017) ? ((f30 > 0.06450012) ? -0.15551404644361 : ((f29 > 1.5) ? 0.188743178704554 : -0.0657985518562245)) : ((f27 > 0.0703292) ? ((f6 > -0.1296669) ? ((f1 > 0.3394569) ? ((f25 > 0.2654597) ? ((f28 > 0.08436333) ? -0.313283248639326 : ((f31 > 12.5) ? ((f25 > 0.365082) ? 0.0470529187965055 : 0.583923107431344) : -0.141367845063969)) : ((f2 > 0.04748374) ? ((f1 > 0.5533178) ? 0.656707581009002 : ((f31 > 11.5) ? 0.474732736400799 : 0.0177460777984268)) : -0.172126760489544)) : -0.249337735060348) : -0.385816482201098) : ((f32 > 0.03436223) ? ((f6 > -0.1144725) ? ((f29 > 11.5) ? -0.473822642659109 : 0.0721042658117372) : -0.187672233108683) : -0.276102844541954))))) : ((f25 > 1.043569) ? ((f29 > 2.5) ? 0.265194020960498 : -0.032163516018055) : ((f10 > 18.05156) ? ((f7 > -0.1750269) ? ((f29 > 3.5) ? 0.22608652853618 : ((f4 > 1.519449E-05) ? -0.00326451002959053 : -0.243386930060955)) : -0.338339896946931) : -0.395178310116683)));
            double treeOutput11 = ((f32 > 0.04082602) ? ((f9 > 6.384574) ? ((f14 > 0.1550481) ? ((f7 > -0.01966062) ? -0.336025039733619 : ((f10 > 2.620667) ? -0.0316307061484257 : -0.388536288526127)) : ((f29 > 3.5) ? ((f28 > 0.09136954) ? ((f25 > 0.2318654) ? ((f1 > 1.144978) ? 0.11053080400489 : ((f30 > 0.06898096) ? -0.385590637940659 : ((f32 > 0.123332) ? 0.14208391813754 : -0.245838792848927))) : 0.421822223005508) : ((f1 > 0.4380271) ? ((f15 > 0.2532762) ? ((f30 > 0.1189833) ? -0.0756406796778957 : ((f31 > 11.5) ? 0.374026210395147 : 0.184906250862871)) : -0.111827405649746) : -0.111015964860195)) : ((f31 > 5.5) ? ((f20 > 1.363851) ? ((f2 > 0.6819611) ? ((f22 > 0.717842) ? ((f0 > 0.9680058) ? -0.0444520847114802 : 0.435190313417222) : 0.0498894592422687) : -0.0939576012650179) : ((f0 > 0.4850922) ? -0.162466701184052 : 0.00442686468223517)) : -0.223398595427024))) : ((f15 > 0.3815115) ? ((f1 > 0.4223207) ? ((f12 > 0.9221007) ? ((f6 > 0.3372754) ? 0.283081735834774 : ((f25 > 0.2318654) ? ((f28 > 0.09035128) ? -0.105605802837004 : 0.147605317414492) : 0.495395431035283)) : ((f7 > 0.4202187) ? -0.411141408894327 : 0.186443178240344)) : ((f2 > 0.6757982) ? 0.163939564197008 : ((f27 > 0.06083602) ? -0.0912689161968235 : 0.124826017195439))) : ((f31 > 24.5) ? 0.200322958949419 : ((f25 > 0.2708267) ? ((f28 > 0.05799814) ? -0.282029454693342 : -0.0888370330667252) : 0.132069488891538)))) : ((f6 > 0.7000832) ? ((f2 > 0.5437956) ? 0.496262963547113 : -0.107306006626381) : ((f32 > 0.01608692) ? ((f6 > -0.1030164) ? ((f14 > 0.2168038) ? -0.332176111603236 : ((f18 > 4.745362) ? 0.35294803424514 : -0.15815125013385)) : -0.409359130654092) : -0.462355604047242)));
            double treeOutput12 = ((f31 > 10.5) ? ((f1 > 0.9351752) ? ((f30 > 0.09095263) ? ((f18 > 5.619127) ? 0.292379242481346 : ((f27 > 0.5089349) ? -0.728180060491272 : ((f28 > 0.1710655) ? -0.287693711641963 : 0.176968474148395))) : ((f28 > 0.6536086) ? -0.594774785913812 : 0.275114078279841)) : ((f16 > 151.5) ? ((f31 > 23.5) ? ((f30 > 0.009520033) ? ((f4 > 0.03201024) ? 0.195852982936476 : -0.543528451494083) : -0.274813722121009) : -0.232830530638749) : ((f15 > 0.3411807) ? ((f12 > 1.875341) ? ((f21 > 1.349704) ? -0.215210959923316 : ((f6 > -0.1328841) ? ((f25 > 0.2489905) ? ((f3 > 0.3930644) ? 0.245664854329019 : 0.0840156305718686) : ((f1 > 0.2300517) ? 0.630137360488381 : 0.143852501607667)) : -0.07853871508809)) : ((f7 > 0.1550588) ? -0.345534843145664 : 0.080850347060232)) : -0.0957503375098688))) : ((f2 > 1.069078) ? ((f6 > -0.1130137) ? ((f31 > 4.5) ? ((f28 > 0.487019) ? -0.50403066597714 : ((f20 > 2.10298) ? 0.383980203289496 : ((f30 > 0.1378544) ? -0.585882533705884 : 0.131199753148671))) : -0.00351270380268531) : -0.161671599413754) : ((f1 > 1.82885) ? 0.390222380607365 : ((f15 > 0.4564134) ? ((f24 > 0.01213391) ? ((f6 > -0.1030164) ? ((f16 > 96.5) ? ((f10 > 17.93593) ? ((f7 > -0.1686456) ? 0.194526976700349 : -0.105265391607154) : -0.243189039629737) : ((f12 > 0.8754225) ? ((f31 > 5.5) ? ((f27 > 0.04813278) ? ((f1 > 0.704164) ? ((f25 > 0.2911794) ? 0.0925360891852133 : 0.62279228161797) : ((f22 > 0.8395244) ? 0.354604257410349 : -0.0326023870287542)) : 0.235067949104877) : ((f27 > 0.0574194) ? -0.160525583055696 : 0.0288544312325468)) : -0.305848364278748)) : -0.272815405319191) : -0.401141473952357) : -0.335475986258709))));
            double treeOutput13 = ((f1 > 0.5470313) ? ((f6 > 0.4790357) ? ((f9 > 5.534935) ? 0.0502935830608049 : 0.240594378834474) : ((f25 > 0.2760238) ? ((f28 > 0.106931) ? ((f31 > 20.5) ? ((f18 > 3.486244) ? 0.334653156188004 : -0.068410084375471) : ((f16 > 127.5) ? -0.347394347254887 : ((f20 > 1.401759) ? -0.0877543189318433 : -0.27993965788621))) : ((f29 > 2.5) ? ((f14 > 0.1322404) ? ((f25 > 0.4287474) ? ((f7 > -0.03892202) ? -0.204011879262781 : ((f10 > 2.858199) ? 0.148063651547557 : -0.201989063738329)) : ((f31 > 9.5) ? 0.782841688818006 : 0.0203209353939804)) : ((f7 > -0.260262) ? ((f15 > 0.34851) ? ((f9 > 18.29306) ? 0.0518639761728124 : 0.23291896385583) : 0.0150269510658129) : -0.0281000156344316)) : ((f29 > 1.5) ? -0.0753770886925172 : -0.27772791485285))) : ((f25 > 0.04795446) ? ((f31 > 5.5) ? 0.511304146577644 : 0.0534520333585556) : ((f30 > 0.0261393) ? 0.273737278914745 : -0.139240157347297)))) : ((f2 > 0.6819611) ? ((f5 > 0.2091092) ? ((f23 > 0.62709) ? 0.00058206817242115 : 0.449487181860373) : ((f25 > 1.258936) ? 0.899909615150613 : ((f0 > 0.5466381) ? ((f23 > 1.952857) ? 0.815155719338451 : ((f26 > 0.6653943) ? ((f4 > 0.004573358) ? 0.740879335781438 : -0.149377348706616) : -0.100206844809046)) : 0.142362301734434))) : ((f2 > 0.0383032) ? ((f6 > -0.08630493) ? ((f14 > 0.4296703) ? -0.29893796684803 : ((f31 > 9.5) ? ((f25 > 0.2708267) ? ((f15 > 0.4459521) ? 0.0343165899271971 : -0.120015067069829) : ((f1 > 0.2639186) ? 0.344174734042926 : 0.0653290676801695)) : ((f27 > 0.06512751) ? -0.208852460465292 : ((f4 > 0.001325623) ? 0.0106390073045711 : -0.204001669174944)))) : -0.310609545622039) : -0.436180255510426)));
            double treeOutput14 = ((f31 > 5.5) ? ((f6 > 0.398509) ? ((f14 > 0.5031645) ? -0.13363803057212 : ((f9 > 8.891764) ? -0.0646588526307275 : ((f1 > 0.4649767) ? ((f5 > 0.1029693) ? ((f25 > 0.2376266) ? ((f1 > 1.173099) ? 0.279681827909115 : ((f30 > 0.06155282) ? -0.30009568883645 : 0.0190358800036601)) : 0.51895007159639) : 0.255165616941614) : ((f2 > 0.56657) ? 0.197516173467634 : ((f26 > 0.1321956) ? -0.0401719281267751 : 0.2329410143867))))) : ((f10 > 16.93398) ? ((f7 > -0.2341071) ? ((f25 > 0.7179731) ? 0.280908887800877 : ((f2 > 0.2446529) ? ((f13 > 0.8038831) ? ((f2 > 0.5437956) ? 0.104861792528005 : ((f31 > 13.5) ? 0.199756890877974 : -0.0726786337310454)) : -0.292377142591784) : 0.367968477621085)) : -0.0741922222527462) : ((f16 > 116.5) ? ((f31 > 19.5) ? ((f30 > 0.01098524) ? ((f25 > 0.4163728) ? -0.0276274910719298 : 0.350265016645306) : -0.244376514857289) : -0.250154576307167) : ((f9 > 11.78133) ? ((f7 > 0.999862) ? ((f25 > 0.7509996) ? 0.627179839450607 : 0.0640717263962366) : -0.18248635954861) : ((f10 > 2.941202) ? ((f1 > 0.7427227) ? 0.277244525603137 : 0.0712009295763112) : ((f12 > 5.600573) ? ((f7 > 0.9524139) ? 0.323281130757646 : ((f15 > 0.5159602) ? 0.0993292862257754 : -0.137124675224141)) : -0.250325628403797)))))) : ((f19 > 5.186791) ? 0.15744901988251 : ((f10 > 17.7122) ? ((f7 > -0.200347) ? ((f24 > 0.008522198) ? ((f32 > 0.04464701) ? -0.0230596571217054 : ((f9 > 6.384574) ? 0.191561335762668 : 0.491331462397306)) : ((f25 > 0.3911977) ? ((f32 > 0.03060906) ? -0.154932020228924 : 1.48014709815466) : -0.270627079712586)) : -0.292372230857266) : ((f7 > 0.9946155) ? -0.0186739024550059 : -0.339894216873296))));
            double treeOutput15 = ((f31 > 12.5) ? ((f16 > 210.5) ? ((f31 > 27.5) ? ((f1 > 0.1797143) ? ((f5 > 4.309524E-05) ? 0.255739285899671 : -0.329630744958297) : -0.381419083546873) : ((f1 > 1.318855) ? 0.192987475826164 : -0.223558566929596)) : ((f15 > 0.4210348) ? ((f12 > 0.8754225) ? ((f30 > 0.1070831) ? ((f1 > 1.428488) ? 0.185199704445244 : -0.295714240574379) : ((f29 > 2.5) ? ((f23 > 0.1890374) ? ((f4 > 0.001325623) ? ((f31 > 17.5) ? 0.430455099922328 : ((f28 > 0.04801884) ? 0.178391227995943 : 0.354886662543488)) : -0.198778043202482) : -0.270810904204795) : 0.0846086919037772)) : ((f3 > 0.6934437) ? 0.485313566068058 : -0.175631627922847)) : ((f11 > 2.525729) ? ((f9 > 9.775102) ? -0.00384498337146046 : 0.343516862518421) : ((f25 > 0.365082) ? ((f18 > 3.803536) ? 0.144461549601382 : ((f30 > 0.05826007) ? -0.380491626067305 : -0.0810215352250889)) : ((f30 > 0.01219075) ? 0.289288140446387 : -0.105166063761082))))) : ((f6 > 0.7404624) ? ((f25 > 0.703735) ? 0.312079373053376 : 0.0830692923714743) : ((f15 > 0.3741958) ? ((f24 > 0.01213391) ? ((f16 > 102.5) ? ((f27 > 0.04066644) ? -0.207122574245016 : -0.0575485061746374) : ((f6 > -0.06719974) ? ((f12 > 0.9221007) ? ((f18 > 2.85683) ? ((f30 > 0.1022561) ? ((f3 > 0.464402) ? 0.149963555700339 : -0.249304643666372) : ((f29 > 1.5) ? 0.284347971959556 : -0.0462597937912314)) : ((f17 > 1.630582) ? ((f7 > -0.2401922) ? 0.0287122995788838 : -0.0988450595965338) : ((f27 > 0.06772219) ? 0.0106363569788077 : ((f31 > 4.5) ? 0.352646761538375 : 0.177403825116009)))) : -0.219256804318897) : ((f6 > -0.1582812) ? -0.0521570083986452 : -0.249055549026404))) : -0.34163319003241) : -0.266954853074036)));
            double treeOutput16 = ((f1 > 0.6444818) ? ((f25 > 0.2862055) ? ((f28 > 0.1156055) ? ((f31 > 14.5) ? ((f1 > 1.094739) ? 0.221651945346143 : ((f29 > 6.5) ? -0.193018790933381 : 0.0797841857803633)) : -0.185934121088335) : ((f29 > 1.5) ? ((f6 > -0.1562471) ? ((f31 > 12.5) ? ((f27 > 0.4001538) ? ((f0 > 1.15797) ? 0.157970717692649 : ((f29 > 8.5) ? -0.758034945732694 : -0.0264809946036453)) : ((f25 > 0.551649) ? ((f21 > 0.04376203) ? ((f23 > 0.2438893) ? 0.285937134535518 : -0.219039297672147) : -0.169720633088293) : 0.593957740605555)) : ((f14 > 0.1680791) ? ((f7 > 0.05854229) ? -0.191744874067364 : ((f16 > 82.5) ? -0.0562723003778084 : 0.282814321078906)) : ((f15 > 0.6481352) ? 0.166171568187134 : 0.0574247696386866))) : -0.129917739248689) : -0.20487269348772)) : ((f19 > 0.1043056) ? ((f31 > 4.5) ? 0.454765749056529 : 0.120927941366496) : 0.00588164862623305)) : ((f32 > 0.1528009) ? ((f28 > 0.202333) ? 0.29004127595136 : ((f3 > 0.5128921) ? ((f25 > 1.258936) ? 0.659568709597517 : -0.0730109182676487) : 0.112809025059271)) : ((f20 > 2.691186) ? ((f27 > 0.004668773) ? ((f2 > 0.8533486) ? 0.926637033476725 : -0.0116510258752803) : -0.0440685149181656) : ((f9 > 5.465028) ? ((f10 > 17.7122) ? ((f7 > -0.2401922) ? ((f31 > 3.5) ? ((f21 > 0.1848973) ? -0.0142065715608374 : ((f3 > 0.1994719) ? 0.0741832191241421 : 0.474051363150507)) : -0.158479789748474) : -0.243562669948571) : ((f7 > 0.965847) ? -0.0203605480298453 : -0.256673536181067)) : ((f32 > 0.02380165) ? ((f27 > 0.07825664) ? -0.0426734015155224 : ((f26 > 0.1860681) ? ((f4 > 1.519449E-05) ? 0.0946149019911323 : -0.115379528276161) : 0.256690862035533)) : -0.251245358055803)))));
            double treeOutput17 = ((f32 > 0.02739079) ? ((f9 > 9.233226) ? ((f6 > -0.1582812) ? ((f18 > 6.742136) ? ((f4 > 0.872744) ? -0.142319920289718 : 0.451341063822945) : ((f30 > 0.09833287) ? -0.205096208851967 : ((f14 > 0.106203) ? ((f7 > 0.01405178) ? -0.232841377884502 : -0.0247883175932039) : ((f29 > 3.5) ? ((f16 > 117.5) ? -0.0333576029795582 : ((f13 > 0.8539063) ? ((f1 > 0.9351752) ? 0.260078101356048 : ((f27 > 0.1348027) ? ((f30 > 0.0571638) ? -0.218772001201081 : 0.0840792229320938) : 0.16703124994326)) : -0.387822978218259)) : ((f23 > 0.4093622) ? -0.0645805838599955 : ((f28 > 0.003329541) ? ((f2 > 0.5781242) ? 0.27403750454236 : 0.0524512446629417) : 0.00187706546401623)))))) : -0.215246987676788) : ((f10 > 2.782951) ? ((f7 > -0.2200399) ? ((f7 > 0.0286411) ? ((f12 > 3.82842) ? 0.154966646118742 : -0.16039849883311) : ((f31 > 4.5) ? ((f19 > 1.825205) ? ((f14 > 0.4833148) ? 0.492864001101524 : 0.211657960662848) : ((f21 > 0.1816684) ? 0.0783629751639872 : ((f26 > 0.1644902) ? 0.140885547121666 : 0.568072346226381))) : 0.0759222416114031)) : ((f31 > 15.5) ? 0.183822521018211 : ((f10 > 5.13386) ? ((f10 > 18.05156) ? -0.0216255215907966 : 0.121202819133324) : -0.13651234860425))) : ((f12 > 5.691823) ? ((f7 > 0.9524139) ? 0.271914967359924 : ((f6 > 0.6505057) ? 0.211800543287579 : ((f13 > 0.02541088) ? ((f6 > 0.01872206) ? ((f31 > 27.5) ? 0.19148149228767 : ((f16 > 187.5) ? -0.176497376553092 : 0.0152688924930963)) : -0.183169888487632) : 0.851464710761275))) : ((f10 > 1.522427) ? -0.14663574147312 : -0.373424376836109)))) : ((f2 > 0.05502445) ? ((f6 > 0.3669006) ? 0.0911007357385182 : -0.188223579055347) : -0.381071404718568));
            double treeOutput18 = ((f1 > 1.428488) ? ((f27 > 0.7585866) ? ((f1 > 2.366133) ? 0.0271704904161807 : -0.492998007061728) : ((f30 > 0.153958) ? ((f26 > 0.2467872) ? ((f18 > 6.742136) ? 0.431151182151248 : -0.230718574477475) : -0.690104782270839) : ((f18 > 6.075307) ? 0.436950826750065 : 0.156340699301649))) : ((f31 > 4.5) ? ((f27 > 0.05571678) ? ((f2 > 0.9597236) ? ((f14 > 0.110101) ? -0.0233749418501516 : 0.222781456120814) : ((f30 > 0.06755489) ? ((f2 > 0.2377282) ? ((f27 > 0.1171723) ? -0.264475755238735 : 0.0510619408642159) : 0.38086048304804) : ((f31 > 15.5) ? ((f16 > 276.5) ? ((f29 > 11.5) ? 0.101169745194576 : -0.252287401950846) : ((f1 > 0.2300517) ? ((f23 > 0.2197619) ? ((f31 > 24.5) ? 0.476241125642484 : ((f25 > 0.4122096) ? ((f15 > 0.2960026) ? 0.158042943682513 : -0.0727692928191142) : 0.358948887214376)) : -0.218978177771769) : -0.0617968120251412)) : ((f16 > 118.5) ? ((f26 > 0.752671) ? -0.55963272330325 : -0.201383869306733) : ((f1 > 0.7968202) ? ((f25 > 0.4080476) ? ((f26 > 0.6896279) ? -0.375881000719962 : ((f12 > 0.9221007) ? 0.108517878660248 : -0.315661089838967)) : 0.335900287331122) : ((f32 > 0.2305679) ? 0.165271800473325 : ((f5 > -0.0004794692) ? -0.0571156752634368 : -0.389299712163793))))))) : ((f16 > 140.5) ? -0.0753881651381561 : ((f15 > 0.2572331) ? ((f29 > 3.5) ? ((f4 > 0.001325623) ? 0.341949334954581 : ((f25 > 0.6395183) ? -0.0357603464260201 : ((f18 > 1.85924) ? 0.44859006123577 : 0.0737994827783079))) : ((f26 > 0.1463143) ? ((f31 > 8.5) ? 0.176925275321579 : 0.00557845408025983) : ((f6 > -0.1448343) ? 0.322589263450657 : -0.0802351207764603))) : -0.056297645143784))) : -0.209977796634466));
            double treeOutput19 = ((f1 > 0.02280172) ? ((f6 > 0.622574) ? ((f9 > 3.823297) ? 0.0466876358053466 : 0.18957553593447) : ((f14 > 0.2877847) ? ((f10 > 15.54926) ? ((f31 > 8.5) ? 0.395993669867966 : 0.160415044748681) : ((f16 > 154.5) ? -0.298778716548217 : ((f7 > 0.1055546) ? -0.206527970273444 : ((f13 > 1.342355) ? -0.113680708092955 : ((f32 > 0.08836708) ? 0.328332550139004 : 0.0575782564761775))))) : ((f7 > 0.9946155) ? 0.223318776872676 : ((f15 > 0.2727511) ? ((f29 > 3.5) ? ((f30 > 0.07687962) ? ((f1 > 1.094739) ? ((f30 > 0.1378544) ? ((f1 > 2.366133) ? 0.299055267414202 : ((f25 > 1.450612) ? -0.58983733713885 : ((f27 > 0.6153364) ? -0.747250117442175 : -0.0757501214197359))) : 0.156501923633045) : ((f27 > 0.1670926) ? -0.310222541092319 : -0.00799551812082225)) : ((f1 > 0.45953) ? ((f16 > 237.5) ? ((f29 > 10.5) ? 0.222934746380114 : -0.115051256096336) : ((f30 > 0.04939323) ? ((f1 > 0.8465301) ? 0.249951918843402 : ((f27 > 0.1485165) ? -0.167839584489983 : 0.0918267215489374)) : 0.284519811533541)) : -0.0117549002958703)) : ((f23 > 0.3326399) ? ((f25 > 0.703735) ? ((f4 > 1.519449E-05) ? ((f26 > 0.3794239) ? ((f24 > 0.5701032) ? 0.0171392699418531 : ((f3 > 0.7317827) ? 0.564409624151737 : 0.224056222319926)) : -0.0131797736792566) : -0.0519887752646017) : -0.0636405714499311) : ((f0 > 0.7899315) ? ((f26 > 0.5369684) ? 0.602645862676131 : -0.214704424298134) : ((f6 > -0.1234417) ? ((f31 > 8.5) ? 0.248744392293867 : ((f27 > 0.06512751) ? -0.0163179500281934 : 0.284840234447065)) : -0.0408236101668516)))) : ((f30 > 0.04542997) ? -0.22586596055254 : -0.102362851628631))))) : ((f25 > 1.258936) ? 1.05803003840067 : -0.285056447051804));
            double treeOutput20 = ((f31 > 11.5) ? ((f10 > 2.398641) ? ((f9 > 11.24498) ? 0.00425484339379329 : ((f16 > 127.5) ? ((f31 > 20.5) ? 0.166848751757987 : ((f22 > 0.2799896) ? ((f25 > 0.2082298) ? -0.212211682815297 : 0.257874978363557) : 0.070964855913481)) : ((f4 > 1.519449E-05) ? ((f27 > 0.3349689) ? ((f0 > 1.22584) ? 0.246490600292436 : -0.149220748515889) : ((f18 > 0.4175414) ? ((f23 > 0.1560663) ? 0.255444902909945 : -0.0598403815557096) : 0.0635066701909219)) : -0.247727718991259))) : ((f12 > 4.697552) ? ((f7 > 0.8778428) ? 0.22504429143967 : ((f9 > 4.037152) ? -0.101040449291177 : ((f31 > 21.5) ? 0.19977270950341 : 0.00547816249244346))) : -0.181852234982169)) : ((f15 > 0.5999436) ? ((f25 > 0.6779889) ? ((f9 > 5.186881) ? ((f26 > 0.3865668) ? ((f0 > 1.65807) ? ((f4 > 0.872744) ? -0.26757053005472 : 0.345707624459101) : ((f23 > 0.6328585) ? -0.0205546190843218 : 0.205900695657098)) : ((f26 > 0.1225576) ? -0.042994359947754 : 0.312365186889494)) : 0.180994982712417) : ((f6 > -0.1044385) ? ((f16 > 83.5) ? -0.0863444501345364 : ((f31 > 2.5) ? ((f21 > 0.1465017) ? ((f25 > 0.1180206) ? ((f17 > 2.011431) ? -0.0553478995914784 : ((f25 > 0.4369328) ? ((f1 > 0.4275262) ? 0.0513455270739138 : 0.273922627133512) : -0.0368440001900039)) : ((f5 > 0.005724818) ? 0.423503335338054 : 0.0182109106515431)) : ((f3 > 0.2045506) ? ((f29 > 3.5) ? 0.25814924982838 : 0.0344662647694932) : ((f9 > 5.67567) ? 0.306280382494875 : 0.549962883886953))) : -0.0831093192325264)) : -0.205423469496232)) : ((f16 > 82.5) ? -0.203200556091263 : ((f6 > 0.1391294) ? ((f1 > 0.01355108) ? 0.0373693421586718 : -0.253655957846371) : -0.142960792560232))));
            double treeOutput21 = ((f18 > 3.414886) ? ((f25 > 0.3954515) ? ((f1 > 2.029585) ? ((f21 > 2.898454) ? -0.0663803073228395 : 0.342445946947474) : ((f30 > 0.1189833) ? -0.199530706129265 : ((f26 > 0.752671) ? ((f32 > 0.1363017) ? 0.209159631060118 : -0.205442591864903) : ((f14 > 0.4217939) ? -0.0893698597497112 : ((f2 > 0.5781242) ? ((f31 > 4.5) ? 0.209133011595277 : 0.00927973719911092) : ((f17 > 5.061989) ? -0.367575140805029 : ((f31 > 6.5) ? 0.46571720111174 : -0.142482660958254))))))) : 0.357671119363701) : ((f2 > 0.03139912) ? ((f6 > -0.1466458) ? ((f30 > 0.05826007) ? ((f27 > 0.1485165) ? -0.235150196869754 : -0.0331993378340772) : ((f31 > 8.5) ? ((f25 > 0.2545401) ? ((f28 > 0.05146447) ? ((f32 > 0.1061548) ? 0.0642513908191688 : -0.101699206750875) : ((f1 > 0.4327258) ? ((f25 > 0.4530378) ? ((f26 > 0.282773) ? ((f4 > 1.519449E-05) ? 0.142557326931707 : -0.175549051962297) : ((f29 > 4.5) ? 0.300606788654995 : -0.158598945332239)) : ((f27 > 0.1960193) ? 0.0786987429735511 : 0.594513427999393)) : ((f26 > 0.5001454) ? 0.395914014594738 : ((f23 > 0.258561) ? -0.0341819826280222 : 0.177031862485276)))) : ((f1 > 0.4069912) ? ((f2 > 0.09828465) ? 0.592334504456012 : 0.315922928124078) : ((f2 > 0.09005587) ? 0.194123608894051 : -0.0857479070128695))) : ((f7 > -0.260262) ? ((f10 > 16.84369) ? ((f25 > 1.158015) ? 0.52521242278823 : ((f29 > 4.5) ? 0.289538317832653 : ((f32 > 0.05404603) ? -0.0125350601228002 : ((f19 > 1.732632) ? ((f7 > -0.1691514) ? 0.470339305379594 : 0.154027702463296) : 0.115834449734654)))) : ((f7 > 0.9994357) ? 0.185696000152527 : -0.0602617065107092)) : -0.121770550208059))) : -0.205226824206696) : -0.32047856460406));
            double treeOutput22 = ((f29 > 3.5) ? ((f28 > 0.08244063) ? ((f25 > 0.2022738) ? ((f1 > 0.8465301) ? ((f31 > 17.5) ? 0.185879223623783 : ((f25 > 1.450612) ? -0.550570610572459 : ((f26 > 0.244778) ? ((f10 > 10.31558) ? ((f3 > 0.4219701) ? 0.145428557275724 : ((f0 > 1.22584) ? -0.270683947531174 : 0.290274539944332)) : -0.147977831536869) : 0.482002914422794))) : ((f19 > 3.139179) ? 0.388840519049712 : -0.204739721722537)) : 0.308390421960515) : ((f5 > 4.309524E-05) ? ((f18 > 1.385967) ? ((f23 > 0.2438893) ? ((f2 > 0.6462063) ? ((f28 > 0.03100014) ? 0.0682092090824249 : 0.269053693664144) : ((f2 > 0.1026826) ? ((f27 > 0.2497463) ? 0.263872037529334 : ((f1 > 0.504689) ? 0.622524324220208 : 0.349901850320664)) : 0.253670422776923)) : -0.216305786113655) : 0.0230917562371162) : ((f16 > 96.5) ? ((f32 > 0.09735045) ? 0.080221101328015 : -0.110550446615722) : ((f29 > 7.5) ? -0.0984492409751963 : ((f18 > 1.352096) ? 0.164028307277284 : -0.0441237371376871))))) : ((f6 > 0.8226977) ? 0.166972843957041 : ((f15 > 0.3187227) ? ((f5 > 0.00263009) ? ((f28 > 0.03263234) ? ((f25 > 0.07403839) ? ((f32 > 0.1935912) ? ((f0 > 0.7303274) ? -0.0245843472194195 : 0.303297090242565) : ((f31 > 9.5) ? 0.0198763549097073 : -0.128260931224367)) : ((f16 > 87.5) ? 0.0158774197868971 : 0.333316601410502)) : ((f16 > 125.5) ? -0.0255441898283702 : ((f2 > 0.2721171) ? 0.107199225211413 : ((f1 > 0.05823568) ? 0.672538695411538 : 0.0768716638975911)))) : ((f25 > 0.05826411) ? ((f23 > 0.234156) ? ((f3 > 0.09560426) ? -0.0562802832031455 : 0.170508156969133) : ((f2 > 0.9313065) ? 0.291823858710779 : 0.0857475684157124)) : -0.251781422021662)) : -0.195914765889719)));
            double treeOutput23 = ((f18 > 5.26422) ? ((f21 > 2.898454) ? ((f1 > 2.366133) ? ((f23 > 1.952857) ? -0.166533948464511 : 0.419661770194585) : -0.39144480466898) : ((f3 > 2.29223E-05) ? 0.219986997474983 : -0.323693730085151)) : ((f6 > -0.1130137) ? ((f15 > 0.6547681) ? ((f31 > 3.5) ? ((f14 > 0.5031645) ? ((f10 > 14.09978) ? 0.418119662107947 : -0.136244597218804) : ((f31 > 13.5) ? ((f16 > 451.5) ? -0.158450806113797 : ((f4 > 0.496999) ? -0.183102480532057 : 0.188013025838746)) : ((f27 > 0.03983103) ? ((f25 > 0.6969948) ? 0.121309200383193 : ((f16 > 185.5) ? -0.217722959351254 : ((f5 > -0.0002845971) ? 0.0285758122519921 : -0.305877110831051))) : ((f4 > 0.003057634) ? 0.198002038136419 : 0.0583232354313006)))) : -0.0463641615459397) : ((f30 > 0.07080285) ? ((f27 > 0.4330106) ? -0.394160536743813 : -0.149567651351053) : ((f16 > 90.5) ? ((f31 > 29.5) ? ((f1 > 0.1797143) ? ((f16 > 451.5) ? -0.0982033869301518 : ((f27 > 0.0248323) ? ((f23 > 0.2745172) ? 0.434565363640994 : -0.365169504302702) : -0.573329724142799)) : -0.248941412058532) : ((f27 > 0.08271883) ? -0.126037099870945 : ((f5 > 4.309524E-05) ? ((f29 > 4.5) ? 0.287946804599761 : 0.0341958223553017) : -0.149716656156969))) : ((f29 > 2.5) ? ((f1 > 0.7634702) ? ((f25 > 0.5565051) ? 0.0848815436056028 : 0.295175134935104) : ((f30 > 0.03637309) ? -0.0747702912603139 : ((f1 > 0.3969941) ? ((f15 > 0.2532762) ? 0.281352859669262 : 0.0488810796778) : -0.00130684101223021))) : ((f30 > 0.0254747) ? -0.122290823866158 : ((f1 > 0.01355108) ? ((f23 > 0.4038999) ? -0.0338820115847871 : ((f3 > 0.1263105) ? 0.0441165070135967 : 0.369336880684849)) : -0.195413351202745)))))) : -0.163490267306365));
            double treeOutput24 = ((f32 > 0.01496899) ? ((f14 > 0.1913919) ? ((f10 > 15.74131) ? ((f9 > 11.64751) ? 0.0216405163862642 : 0.252303308922285) : ((f8 > 0.2156714) ? ((f16 > 187.5) ? -0.232303814271589 : ((f9 > 4.712649) ? ((f18 > 6.742136) ? 0.280331546061168 : ((f25 > 0.4996653) ? -0.159602465281486 : ((f18 > 2.016162) ? ((f31 > 11.5) ? 0.436715601164666 : 0.0630125749148512) : -0.154547844593916))) : ((f10 > 1.422399) ? ((f13 > 0.02541088) ? ((f15 > 0.470569) ? ((f18 > 0.9631332) ? ((f31 > 7.5) ? ((f23 > 0.2011348) ? 0.367167442473992 : -0.121973391176099) : 0.160797702630649) : 0.00193703884752177) : -0.0206298733327655) : 0.803487158244816) : -0.276836195099119))) : 0.325143247970306)) : ((f7 > -0.2681015) ? ((f7 > 0.9981551) ? 0.178522793519373 : ((f10 > 2.285497) ? ((f9 > 10.4451) ? ((f10 > 18.4178) ? ((f12 > 0.6421817) ? ((f7 > -0.1680053) ? ((f25 > 0.8813297) ? ((f32 > 0.05404603) ? 0.164813927245144 : 0.736585840085696) : ((f5 > 4.309524E-05) ? 0.145812369054266 : 0.054941193857909)) : ((f9 > 18.29306) ? -0.0793617953061082 : 0.0603992994492091)) : ((f32 > 0.1668385) ? -0.536756035679167 : -0.177143378000809)) : ((f10 > 8.657379) ? -0.0935259961248008 : 0.0505667651486917)) : ((f7 > -0.1714363) ? ((f29 > 3.5) ? ((f30 > 0.07438731) ? 0.0443867216693028 : 0.221909180189362) : ((f23 > 0.4722296) ? 0.0681611827338418 : ((f5 > 0.003619287) ? 0.288107194097116 : ((f25 > 0.8183883) ? 0.422265350838567 : 0.152451108942204)))) : 0.0410695508486853)) : ((f7 > 0.7322611) ? 0.0786747837652056 : ((f31 > 27.5) ? 0.200321037886155 : ((f21 > 0.3117572) ? -0.13127176972881 : -0.0195823118929757))))) : -0.0604313136497187)) : -0.251397521193179);
            double treeOutput25 = ((f31 > 16.5) ? ((f16 > 361.5) ? -0.125155744203323 : ((f27 > 0.008601799) ? ((f18 > 0.4447877) ? ((f23 > 0.2011348) ? ((f27 > 0.2896849) ? 0.0160993305524734 : ((f28 > 0.1750538) ? 0.0329980413120301 : ((f29 > 4.5) ? ((f16 > 145.5) ? 0.226481050106478 : 0.390870777383691) : ((f10 > 2.219203) ? 0.193009821746511 : -0.0083695324724373)))) : ((f14 > 0.110101) ? -0.354966280181975 : 0.231593794431113)) : -0.0878031803078682) : -0.353022205530101)) : ((f16 > 75.5) ? ((f32 > 0.1281802) ? ((f3 > 0.4299013) ? 0.136587653408867 : ((f0 > 1.745745) ? -0.434230806061889 : 0.00377982420892646)) : ((f27 > 0.09837766) ? ((f25 > 0.2201282) ? ((f30 > 0.03577587) ? -0.197839284372007 : -0.120426900991512) : ((f18 > 1.581675) ? ((f2 > 0.0383032) ? ((f3 > 0.1840939) ? 0.442871551733636 : 0.0344799135480385) : 0.00414315363428111) : -0.192533617607323)) : ((f5 > 0.00263009) ? ((f28 > 0.01958055) ? -0.0180008847097329 : 0.22306392862858) : -0.110600517874116))) : ((f18 > 3.000507) ? ((f25 > 0.3606755) ? ((f9 > 18.41715) ? -0.0811608485919497 : ((f29 > 1.5) ? ((f27 > 0.1734888) ? 0.0317966833014305 : 0.184338948962973) : -0.0930533304202919)) : ((f28 > 0.1427906) ? 0.156193071710544 : 0.460835039387866)) : ((f9 > 10.37847) ? -0.0457651701871431 : ((f10 > 2.442347) ? ((f4 > 1.519449E-05) ? ((f4 > 0.04222184) ? ((f31 > 7.5) ? ((f32 > 0.06212721) ? ((f27 > 0.2847115) ? -0.162030621385957 : 0.0469166268902523) : ((f18 > 1.237303) ? 0.436481502945 : 0.182207473869516)) : 0.0106118349718241) : 0.300077423435809) : 0.0100821167059022) : ((f12 > 4.920364) ? ((f7 > 0.8778428) ? 0.140395738388902 : -0.0414015672732185) : -0.252358451096433))))));
            double treeOutput26 = ((f15 > 0.2170031) ? ((f29 > 3.5) ? ((f22 > 0.2799896) ? ((f25 > 0.2022738) ? ((f1 > 0.9351752) ? ((f2 > 1.819301) ? -0.495707684245751 : ((f13 > 0.8539063) ? 0.104842092453402 : -0.170476040741002)) : ((f30 > 0.04939323) ? -0.210905300898934 : -0.0363089399107676)) : 0.256486777298438) : ((f16 > 378.5) ? -0.126861875610178 : ((f30 > 0.08821622) ? ((f27 > 0.4156088) ? ((f1 > 2.029585) ? 0.234931643282104 : -0.527785783897965) : 0.0245819568359015) : ((f1 > 0.3724392) ? ((f25 > 0.5904019) ? ((f0 > 1.455858) ? 0.255355862386928 : ((f4 > 1.519449E-05) ? ((f32 > 0.1399797) ? 0.377240132435889 : 0.0558779223043261) : -0.114429234234395)) : ((f27 > 0.1511681) ? ((f5 > 4.309524E-05) ? 0.248292771101631 : -0.0499350066407007) : ((f31 > 9.5) ? 0.397657468309274 : ((f16 > 190.5) ? -0.0630155178158391 : 0.279980535136112)))) : -0.00401681384335079)))) : ((f25 > 0.7255468) ? ((f23 > 0.5664996) ? ((f0 > 1.499442) ? ((f27 > 0.6153364) ? -0.214055134482822 : ((f0 > 2.36744) ? 0.485181526420577 : ((f27 > 0.003365818) ? 0.23482742356983 : -0.0290608967646707))) : -0.0297137842453178) : ((f25 > 0.9464693) ? ((f0 > 0.6963166) ? 0.623404157929083 : 0.215385536123951) : 0.123328311702307)) : ((f30 > 0.04080816) ? -0.106072329161075 : ((f5 > 0.003619287) ? ((f27 > 0.03415264) ? ((f16 > 105.5) ? -0.0937426773314069 : ((f2 > 0.1073311) ? ((f23 > 0.6101959) ? -0.0623477285181807 : ((f20 > 1.149059) ? ((f25 > 0.4732654) ? 0.229667432599182 : 0.0158877982262379) : -0.0167242609860128)) : 0.242803785279078)) : ((f1 > 0.04488352) ? 0.33132146017017 : 0.0245326388855008)) : -0.0605319527718234)))) : ((f32 > 0.05562594) ? -0.155268076506247 : -0.138865081426651));
            double treeOutput27 = ((f31 > 18.5) ? ((f27 > 0.01329084) ? ((f16 > 289.5) ? -0.0387122239768317 : ((f1 > 0.1504544) ? ((f23 > 0.2536597) ? ((f27 > 0.2896849) ? 0.0473533480276502 : 0.228296403407627) : -0.151059780696307) : -0.052694628488973)) : -0.333550970574214) : ((f16 > 127.5) ? ((f10 > 18.4178) ? ((f7 > -0.1680053) ? 0.189678691681587 : -0.00977139527467323) : ((f7 > 0.9994357) ? 0.327528597300719 : ((f27 > 0.07120473) ? -0.176446602315852 : -0.124006044391028))) : ((f31 > 5.5) ? ((f11 > 4.912182) ? ((f9 > 12.18369) ? 0.0626596598582677 : ((f32 > 0.05562594) ? 0.0726012520549708 : ((f1 > 0.4873957) ? 0.554480693756007 : 0.321526234303114))) : ((f27 > 0.04979695) ? ((f15 > 0.2046498) ? ((f12 > 0.8754225) ? ((f18 > 1.85924) ? ((f29 > 1.5) ? ((f30 > 0.05617291) ? ((f1 > 0.9880317) ? 0.0732455979066293 : ((f4 > 0.3862948) ? -0.332777238310673 : -0.0257251994019107)) : ((f25 > 0.5088348) ? ((f0 > 1.396596) ? ((f3 > 0.3833796) ? 0.315640290380823 : -0.0784090746276726) : 0.0287458244123817) : 0.267246142114599)) : -0.172177570976159) : ((f30 > 0.03060765) ? -0.0995428152876448 : ((f26 > 0.5170648) ? 0.329989001389385 : ((f22 > 0.6700493) ? 0.19458177092366 : ((f23 > 0.258561) ? ((f1 > 0.2469554) ? ((f2 > 0.3285598) ? ((f29 > 1.5) ? 0.0317137738836565 : -0.168468277556343) : ((f31 > 9.5) ? 0.469360379667076 : 0.127394976421915)) : -0.10481895034891) : 0.12085016893209))))) : ((f26 > 0.6056879) ? 0.346770806216811 : -0.166735048634863)) : -0.155628168416974) : ((f26 > 0.1622604) ? ((f1 > 1.144978) ? 0.257270571100301 : 0.0178689664925898) : 0.210904482076652))) : ((f6 > -0.1265307) ? -0.0194048591424003 : -0.250852449379756))));
            double treeOutput28 = ((f15 > 0.5581526) ? ((f31 > 9.5) ? ((f16 > 236.5) ? -0.0339314913254323 : ((f32 > 0.05562594) ? ((f31 > 17.5) ? 0.163713281826889 : ((f27 > 0.2110251) ? ((f9 > 18.41715) ? -0.152272272430471 : 0.0203771410481846) : ((f20 > 1.339947) ? ((f2 > 0.4715587) ? 0.150980187532901 : ((f5 > 0.156965) ? 0.81948812899741 : 0.355061914153954)) : ((f17 > 5.636498) ? -0.365984587682177 : ((f2 > 0.2377282) ? 0.0110702111154135 : 0.251300838053235))))) : ((f9 > 5.9577) ? 0.184077874055648 : ((f1 > 0.217474) ? 0.482280314252099 : 0.183441773348864)))) : ((f2 > 0.5437956) ? ((f30 > 0.01265378) ? ((f0 > 0.6211545) ? ((f29 > 1.5) ? ((f27 > 0.02782674) ? ((f30 > 0.02512359) ? -0.0417530406600123 : 0.129398659701132) : ((f4 > 1.519449E-05) ? ((f29 > 4.5) ? 0.465931126120585 : 0.133220215261929) : 0.0457932595097284)) : -0.112704178127385) : ((f20 > 1.721656) ? -0.0592610864818559 : 0.148470198037986)) : ((f16 > 86.5) ? 0.0606261563850069 : ((f32 > 0.02076208) ? 0.242920819948561 : ((f1 > 0.001459654) ? 0.773429259429234 : 3.05421399643429)))) : ((f5 > -0.0002845971) ? -0.0306468500397107 : -0.278434085602937))) : ((f13 > 0.02541088) ? ((f14 > 0.1526807) ? ((f25 > 0.551649) ? -0.155383188158437 : ((f18 > 1.715591) ? ((f31 > 7.5) ? 0.160834029430618 : -0.107384950206795) : -0.182713239741283)) : ((f30 > 0.05126864) ? -0.0909853549400949 : ((f29 > 3.5) ? ((f18 > 1.65724) ? ((f11 > 4.457438) ? 0.262365082634148 : ((f15 > 0.1525831) ? 0.119867194866152 : -0.0526117634649892)) : -0.00533950075956869) : ((f25 > 0.9008789) ? 0.128348193109484 : ((f20 > 3.100752) ? -0.33355104815083 : -0.0505883535121983))))) : 0.487177414625981));
            double treeOutput29 = ((f18 > 5.619127) ? ((f21 > 2.898454) ? -0.0995240616346294 : ((f26 > 0.1839525) ? 0.249421965542406 : ((f30 > 0.1120599) ? -0.489432215243722 : 0.311315469422566))) : ((f2 > 0.02664452) ? ((f14 > 0.5031645) ? ((f13 > 1.163744) ? -0.260334382721676 : ((f7 > 0.07401344) ? ((f9 > 17.36975) ? ((f26 > 0.5266525) ? 1.00702596386866 : 0.138451433075118) : -0.224861825237765) : ((f12 > 1.128524) ? -0.0398042620114859 : ((f32 > 0.07828732) ? ((f9 > 18.41715) ? -0.0455215792205593 : 0.563875809254076) : 0.28825903840803)))) : ((f13 > 0.02541088) ? ((f6 > 0.09825749) ? ((f9 > 7.994934) ? -0.0111549818935459 : ((f32 > 0.06024682) ? ((f12 > 9.054589) ? ((f30 > 0.08821622) ? -0.232506308686036 : ((f29 > 3.5) ? 0.037970135339717 : ((f23 > 0.3932374) ? -0.0961767635037832 : 0.0501567606954305))) : ((f32 > 0.123332) ? 0.122235150094147 : 0.0141437828874511)) : ((f10 > 2.89904) ? ((f2 > 0.5437956) ? 0.245802289139743 : ((f28 > 4.172743E-05) ? 0.214965649036477 : 0.124511689376628)) : ((f7 > 0.9878271) ? 0.420180617443839 : ((f11 > 2.936746) ? ((f31 > 8.5) ? 0.321377509882222 : 0.126224113443858) : 0.0150006973033553))))) : ((f7 > -0.3162376) ? ((f10 > 16.66873) ? ((f25 > 1.092159) ? 0.300732509877147 : ((f7 > -0.1686456) ? ((f13 > 0.8038831) ? 0.0908414098329979 : -0.168739641200174) : ((f9 > 18.29306) ? ((f30 > 0.04764608) ? -0.204835870944758 : -0.0716188503445929) : 0.0529476047672262))) : ((f7 > 0.923084) ? 0.0917198909410796 : ((f10 > 2.285497) ? ((f9 > 14.28095) ? -0.114174382911788 : ((f31 > 9.5) ? 0.127415229308648 : -0.00602163851376918)) : -0.148117971835567))) : -0.124466955262716)) : 0.626881101940874)) : -0.249370456373595));
            double treeOutput30 = ((f6 > 0.7640513) ? ((f9 > 4.517768) ? -0.0150081651410324 : 0.151415752735752) : ((f16 > 61.5) ? ((f32 > 0.1251566) ? ((f30 > 0.1120599) ? ((f1 > 2.366133) ? 0.19926946882303 : -0.159957388975744) : ((f3 > 0.4105584) ? ((f27 > 0.001190391) ? 0.121985267513637 : ((f24 > 0.6097875) ? 0.356745076286783 : -0.11016006528008)) : ((f27 > 0.01600741) ? -0.0383260071935697 : 0.334341470925952))) : ((f30 > 0.05617291) ? ((f1 > 1.203706) ? ((f21 > 1.936216) ? ((f1 > 1.82885) ? -0.0211678462368655 : -0.551959069795991) : 0.0924541340885323) : ((f26 > 0.08921942) ? ((f27 > 0.3742904) ? -0.450684895286231 : -0.15761188982438) : 0.621759338544848)) : ((f29 > 4.5) ? ((f16 > 451.5) ? ((f29 > 15.5) ? 0.105454917538883 : -0.308500251909915) : ((f23 > 0.1849836) ? ((f1 > 0.6214421) ? ((f15 > 0.2572331) ? ((f16 > 149.5) ? 0.140289169263007 : 0.291018535913793) : ((f12 > 22.43898) ? 0.20338371164123 : -0.0837901358068284)) : ((f31 > 32.5) ? 0.349058440995237 : ((f26 > 0.2111623) ? -0.0637862348082934 : 0.132573181426989))) : -0.316684268074526)) : ((f30 > 0.04080816) ? -0.11682908451287 : ((f16 > 148.5) ? -0.174121482383592 : ((f31 > 9.5) ? ((f27 > 0.04396158) ? ((f15 > 0.6583485) ? 0.100831441730927 : -0.014397911481591) : 0.194347844522683) : -0.0596040624624473)))))) : ((f29 > 2.5) ? ((f12 > 1.875341) ? 0.105290683855197 : ((f20 > 2.025493) ? 0.222667140567654 : -0.101888250898946)) : ((f30 > 0.03390822) ? -0.0888187654384545 : ((f31 > 2.5) ? ((f23 > 0.2635554) ? 0.0132037520204379 : ((f0 > 0.6908955) ? -0.104758394684995 : ((f25 > 0.4080476) ? 0.293367434477363 : 0.185169273838214))) : -0.0828071320398425)))));
            double treeOutput31 = ((f6 > -0.1791604) ? ((f24 > 0.008522198) ? ((f26 > 0.09856266) ? ((f11 > 7.259601) ? ((f5 > 4.309524E-05) ? 0.154093946675731 : 0.0307960287338126) : ((f10 > 2.513241) ? ((f18 > 3.803536) ? ((f21 > 2.09486) ? ((f1 > 2.029585) ? 0.0947160838965825 : -0.232041479997663) : 0.115416925099163) : ((f30 > 0.07438731) ? -0.124642994313592 : ((f18 > 1.455188) ? ((f25 > 0.310394) ? ((f29 > 1.5) ? ((f23 > 0.2685121) ? ((f30 > 0.04080816) ? ((f18 > 2.370383) ? 0.0432782817684169 : -0.0663498687001233) : ((f15 > 0.1834551) ? ((f29 > 2.5) ? ((f16 > 128.5) ? 0.0387858327077959 : 0.19597327775766) : ((f30 > 0.02482514) ? -0.0226991468827946 : 0.139106690365245)) : -0.0630771682956687)) : ((f28 > 0.02544287) ? -0.256437549154128 : 0.0305568257836508)) : ((f31 > 10.5) ? ((f10 > 16.66873) ? -0.443135206570426 : -0.109866123149905) : -0.0692729327522101)) : ((f31 > 6.5) ? ((f16 > 73.5) ? 0.162074212150736 : 0.394045179115541) : 0.0971969773661382)) : ((f25 > 0.5710722) ? ((f26 > 0.752671) ? ((f4 > 1.519449E-05) ? 0.660342900133056 : 0.113035932442947) : ((f23 > 0.7697505) ? ((f0 > 2.208209) ? 0.519415435541583 : -0.0750727769932059) : 0.0937867924872775)) : ((f30 > 0.02221101) ? -0.0916614007812647 : -0.0257552973062657))))) : ((f12 > 5.509497) ? ((f7 > 0.8178064) ? 0.0765819864266196 : ((f13 > 0.02541088) ? -0.0575511748690482 : 0.473314035892874)) : -0.159994698403884))) : ((f23 > 0.4410134) ? 0.0462385957047189 : ((f32 > 0.02000667) ? ((f21 > 0.1153241) ? ((f31 > 7.5) ? 0.377759781082077 : 0.0853032806581394) : 0.529039238994673) : ((f9 > 2.48424) ? 0.356542514309383 : 1.42604072075647)))) : -0.118759681210553) : -0.188897928248495);
            double treeOutput32 = ((f15 > 0.6930653) ? ((f29 > 2.5) ? ((f30 > 0.0571638) ? ((f18 > 2.53623) ? ((f9 > 18.29306) ? -0.09483343269933 : ((f30 > 0.1070831) ? -0.0248768521526287 : 0.120997612533502)) : -0.0869455790197509) : ((f18 > 1.544705) ? ((f16 > 90.5) ? 0.102130579557743 : ((f30 > 0.04001511) ? ((f1 > 0.7327995) ? ((f13 > 0.9012095) ? 0.308070008379478 : -0.0393444807178248) : 0.0686498139877894) : ((f28 > 0.2075805) ? -0.0405429379628889 : 0.425055681115146))) : ((f2 > 0.6696985) ? ((f30 > 0.0254747) ? 0.0504957094561409 : 0.384757874753267) : ((f3 > 0.117862) ? -0.0360392621220238 : 0.261940856765274)))) : ((f9 > 7.158132) ? -0.019880750630181 : 0.068586386185993)) : ((f16 > 66.5) ? ((f31 > 32.5) ? ((f5 > 4.309524E-05) ? ((f1 > 0.1379996) ? 0.281763002638643 : -0.209032286207906) : -0.285925374216042) : ((f27 > 0.1002947) ? ((f30 > 0.03530023) ? ((f1 > 1.49885) ? ((f25 > 1.00577) ? -0.154502944623331 : 0.181990954065497) : ((f25 > 0.2022738) ? ((f13 > 9.407055) ? -0.221940869643907 : -0.118209743430678) : 0.0773822139059843)) : ((f20 > 1.527422) ? ((f10 > 18.29356) ? ((f23 > 0.6998797) ? ((f32 > 0.03496219) ? ((f29 > 2.5) ? 0.178372931033206 : -0.166734427560253) : 1.14556037149043) : ((f2 > 0.6462063) ? 0.326805678623083 : -0.119049475696987)) : -0.0105696769516514) : -0.0937203786150917)) : ((f29 > 4.5) ? ((f5 > 4.309524E-05) ? 0.183777449960351 : 0.00341110664984841) : -0.0738897873555011))) : ((f31 > 8.5) ? ((f32 > 0.05333724) ? ((f2 > 0.298209) ? 0.00292119174149869 : 0.203273294215922) : ((f6 > 0.06708765) ? ((f15 > 0.1434548) ? 0.38053406284537 : 0.143028576205952) : 0.0583644845625266)) : -0.0130243313084859)));
            double treeOutput33 = ((f31 > 14.5) ? ((f16 > 161.5) ? ((f9 > 4.647275) ? -0.0996303651036371 : ((f31 > 30.5) ? 0.229963923558684 : 0.00543522223929184)) : ((f29 > 11.5) ? ((f5 > 0.009947406) ? -0.00291854124111037 : -0.33098752465253) : ((f18 > 2.53623) ? ((f32 > 0.2425357) ? -0.143592847233466 : 0.25928226956041) : ((f19 > 1.205926) ? ((f26 > 0.313533) ? ((f27 > 0.2756083) ? ((f23 > 0.6444719) ? ((f2 > 0.8533486) ? 0.479735199547908 : -0.289432806707145) : 0.127421913987602) : ((f27 > 0.01735407) ? ((f14 > 0.09600614) ? 0.0443378055551637 : 0.182343805161224) : -0.265518648977358)) : ((f25 > 0.595188) ? -0.212295070036589 : 0.00873461308889269)) : ((f24 > 0.112229) ? ((f2 > 0.0763697) ? ((f30 > 0.02752701) ? 0.179242971807445 : 0.510968400747481) : 0.112080660187655) : -0.059666289072047))))) : ((f7 > -0.2767871) ? ((f7 > -0.1028073) ? ((f12 > 5.691823) ? ((f7 > 0.923084) ? 0.101843099595625 : ((f9 > 7.296778) ? -0.0711752056032875 : ((f10 > 2.14589) ? ((f31 > 3.5) ? ((f16 > 99.5) ? 0.0663179870285504 : 0.217188207021254) : 0.00853762929628767) : -0.0309959128779258))) : -0.149128760880281) : ((f7 > -0.1680053) ? ((f10 > 2.650347) ? ((f10 > 6.273683) ? ((f25 > 0.9223884) ? ((f30 > 0.0133291) ? 0.106457171396752 : 0.487353418192265) : ((f3 > 0.9333973) ? -0.199885612074691 : ((f10 > 17.10492) ? ((f13 > 16.15499) ? -0.0152043990890459 : ((f31 > 0.5) ? 0.0946281267830647 : 0.159828925083514)) : -0.0462711174569892))) : ((f2 > 0.56657) ? ((f10 > 3.238678) ? 0.457859774082052 : 0.115446524179582) : 0.291148767241092)) : -0.175000958312136) : ((f10 > 3.467698) ? 0.016848358390303 : -0.148808387966814))) : -0.088632803528894));
            double treeOutput34 = ((f15 > 0.1480016) ? ((f9 > 4.712649) ? ((f16 > 83.5) ? ((f7 > 0.999862) ? 0.219420822841826 : ((f10 > 18.08576) ? ((f7 > -0.1714363) ? ((f29 > 8.5) ? 0.32644318240066 : 0.0824332277121157) : -0.0378470362927127) : ((f25 > 0.3606755) ? -0.100004895191796 : -0.096688267456293))) : ((f29 > 2.5) ? ((f30 > 0.04542997) ? ((f9 > 18.23667) ? -0.0865819683188733 : ((f18 > 2.401862) ? ((f31 > 11.5) ? 0.211205154268067 : ((f27 > 0.1511681) ? -0.0421895683313447 : 0.0927825371077611)) : -0.0420783683202306)) : ((f18 > 1.352096) ? ((f12 > 2.856128) ? ((f15 > 0.5684311) ? ((f16 > 60.5) ? 0.2304808315446 : 0.462614563375237) : ((f30 > 0.02752701) ? 0.0928793212923338 : 0.340693521000206)) : 0.0597223209454015) : 0.0167196937543754)) : ((f30 > 0.03138225) ? -0.0831278467162876 : ((f31 > 3.5) ? ((f23 > 0.1701278) ? ((f25 > 0.7929175) ? ((f26 > 0.5082138) ? 0.380358072605406 : 0.030646003908446) : ((f2 > 0.2515932) ? ((f29 > 1.5) ? ((f5 > 0.2091092) ? 0.247087476369365 : ((f30 > 0.02181265) ? ((f1 > 0.5661057) ? 0.106042600937513 : -0.0828708070112728) : ((f27 > 0.2034223) ? 0.255334902120031 : 0.0722406425429889))) : ((f30 > 0.01429252) ? ((f32 > 0.1363017) ? ((f1 > 0.1963926) ? -0.327949603467201 : -0.00403299401601125) : -0.157927050285855) : 0.00758126301818565)) : ((f4 > 0.3609461) ? 0.674194398856364 : 0.120410954770545))) : ((f28 > 0.02790445) ? -0.0385542936437845 : ((f0 > 5.913356E-05) ? 0.209168069693215 : 0.430990453080197))) : -0.0695788820329076)))) : ((f6 > 0.1586699) ? ((f31 > 20.5) ? 0.16771674522494 : 0.0622450219512202) : -0.0257758856728281)) : ((f30 > 0.02577597) ? -0.223471108565826 : -0.130045892715259));
            double treeOutput35 = ((f24 > 1.476991) ? 0.191242631488368 : ((f30 > 0.1120599) ? -0.130980328390529 : ((f1 > 1.049464) ? ((f26 > 0.6653943) ? ((f2 > 1.122298) ? 0.0794348794945647 : -0.161386948619593) : ((f28 > 0.1901561) ? ((f4 > 0.3127722) ? 0.167046765756965 : -0.120008757342978) : 0.140114952298187)) : ((f25 > 1.258936) ? ((f31 > 2.5) ? 0.473380572122141 : -0.098165389726091) : ((f30 > 0.05126864) ? ((f26 > 0.02562993) ? ((f32 > 0.1668385) ? 0.0775998073907629 : ((f4 > 0.3688505) ? -0.288336811215726 : -0.0717352757016728)) : 1.20771331242293) : ((f4 > 1.519449E-05) ? ((f27 > 0.03575362) ? ((f18 > 0.99255) ? ((f25 > 0.347221) ? ((f3 > 0.3930644) ? ((f25 > 0.684077) ? ((f14 > 0.08989899) ? 0.00221135551397736 : 0.15516942980637) : ((f4 > 0.288074) ? ((f17 > 4.639391) ? 0.228685130210544 : ((f32 > 0.1363017) ? -0.269885420511328 : -0.139243523715587)) : ((f3 > 0.7803613) ? -0.394899133385321 : 0.0463239740401581))) : ((f4 > 0.09908597) ? -0.0916145336133721 : ((f0 > 0.44257) ? -0.00946218879219464 : 0.318522467282006))) : ((f31 > 8.5) ? ((f2 > 0.0763697) ? ((f28 > 0.04338297) ? 0.174581367842249 : 0.456143443747386) : 0.0165665244855829) : 0.0353969589636928)) : ((f3 > 0.7544315) ? 0.481525925415706 : ((f30 > 0.01999001) ? -0.130797510576073 : ((f2 > 0.5215745) ? ((f23 > 0.5611383) ? -0.0483630118423725 : 0.151484270954508) : -0.0879195590453784)))) : ((f19 > 1.862985) ? 0.179884412417457 : 0.142161886772611)) : ((f3 > 0.4219701) ? -0.0948577095370877 : ((f24 > 0.01213391) ? ((f24 > 0.09806815) ? -0.0119493645189821 : 0.269575604372554) : ((f19 > 1.843888) ? ((f32 > 0.02285403) ? 0.0785068379495771 : 1.17193639549797) : -0.192361849283068)))))))));
            double treeOutput36 = ((f15 > 0.53378) ? ((f29 > 3.5) ? ((f30 > 0.07687962) ? ((f1 > 2.366133) ? 0.314944180496888 : ((f30 > 0.1850781) ? -0.266854834108886 : ((f18 > 4.236958) ? 0.0688927919041393 : ((f26 > 0.1321956) ? ((f16 > 97.5) ? -0.230253027427241 : -0.0305684919629672) : 0.547730518855827)))) : ((f16 > 438.5) ? -0.105118989577785 : ((f18 > 2.434362) ? ((f2 > 0.7066675) ? ((f0 > 1.499442) ? 0.23148935877317 : 0.0484040653982615) : ((f27 > 0.01396389) ? 0.235436937284552 : ((f3 > 0.3713172) ? 0.656015104021324 : 0.168471170589014))) : ((f23 > 0.1921333) ? ((f30 > 0.05126864) ? -0.0236903796630407 : ((f18 > 1.269553) ? 0.148790889803624 : 0.00518332186042881)) : -0.261867796636586)))) : ((f30 > 0.03446827) ? ((f16 > 68.5) ? -0.123012164954319 : ((f1 > 0.9695498) ? 0.0839446563169099 : -0.0386053688938267)) : ((f5 > 4.309524E-05) ? ((f28 > 0.02402961) ? ((f18 > 0.555415) ? ((f23 > 0.6444719) ? -0.000386186992544202 : ((f3 > 0.3486301) ? 0.167770457192952 : 0.0730631675518294)) : ((f4 > 0.02100587) ? ((f28 > 0.1427906) ? 0.176133781686203 : -0.111454905143101) : 1.64832252833779)) : 0.146727007248037) : -0.000728018749312984))) : ((f22 > 0.2387211) ? ((f2 > 0.2096379) ? ((f31 > 30.5) ? 0.179173218234156 : -0.108579056506222) : 0.0845747309561167) : ((f28 > 0.002456723) ? ((f24 > 0.5701032) ? ((f25 > 0.6243948) ? 0.109601544133372 : 0.380788054108604) : ((f18 > 0.3212532) ? ((f25 > 0.3996871) ? -0.0160938826531611 : ((f1 > 0.504689) ? ((f25 > 0.1088893) ? 0.567890585877453 : 0.146981635967863) : ((f26 > 0.2271789) ? -0.0746161590775534 : 0.179696376435878))) : -0.100596512019354)) : ((f27 > 0.09647223) ? -0.0883308714397946 : -0.0457354813554743))));
            double treeOutput37 = ((f31 > 19.5) ? ((f27 > 0.01735407) ? ((f16 > 426.5) ? -0.0948296433024354 : ((f1 > 0.1091233) ? 0.128967679142899 : -0.0734737715739994)) : -0.297306805607241) : ((f16 > 102.5) ? ((f27 > 0.06426305) ? ((f32 > 0.2798601) ? 0.278237099122323 : ((f30 > 0.02221101) ? ((f25 > 0.1902274) ? -0.12662762630482 : 0.0400671866413468) : -0.0712777849566872)) : ((f25 > 1.158015) ? 0.190459846218089 : -0.0406148863140444)) : ((f31 > 9.5) ? ((f27 > 0.2190944) ? ((f18 > 2.85683) ? ((f29 > 7.5) ? -0.119606069954469 : 0.130283372438146) : ((f0 > 1.255128) ? ((f28 > 0.07962343) ? -0.0442274132350719 : 0.356263012538646) : ((f1 > 0.1630143) ? ((f2 > 0.3044708) ? -0.155190881873694 : 0.216082568799595) : 0.130974252490217))) : ((f10 > 1.733314) ? ((f4 > 1.519449E-05) ? ((f29 > 2.5) ? ((f5 > 0.04535082) ? ((f23 > 0.2150494) ? ((f1 > 0.4988552) ? ((f17 > 5.636498) ? -0.122764152105431 : 0.171812413889909) : -0.0265051471226251) : -0.320141819212628) : 0.207878081404871) : ((f23 > 0.4038999) ? ((f32 > 0.0989999) ? -0.113725783523744 : 0.0294007764720785) : ((f24 > 0.05856755) ? ((f0 > 0.9680058) ? -0.135183989346774 : 0.247239540333992) : 0.0452401928395898))) : -0.134121141564434) : -0.0410487710891055)) : ((f28 > -0.0002392505) ? ((f15 > 0.2806952) ? ((f9 > 8.549267) ? -0.00715817117663315 : ((f32 > 0.06328329) ? -0.0353533846268563 : ((f31 > 3.5) ? ((f1 > 0.8739062) ? 0.251479387375059 : ((f26 > 0.07962061) ? ((f2 > 0.5608138) ? 0.147426297608218 : ((f18 > 1.526607) ? 0.191633206008363 : 0.0179338919438219)) : 0.276180767545755)) : ((f28 > 0.003329541) ? 0.556911778699034 : 0.0549621615512924)))) : -0.0945975512469662) : -0.233953585779391))));
            double treeOutput38 = ((f7 > -0.2681015) ? ((f7 > -0.07501728) ? ((f12 > 4.854751) ? ((f13 > 0.02541088) ? ((f7 > 0.9524139) ? 0.088665793313266 : ((f6 > 0.7911065) ? 0.189949149108467 : ((f10 > 10.72248) ? -0.115706902341793 : ((f10 > 2.053907) ? ((f31 > 7.5) ? ((f16 > 94.5) ? ((f9 > 7.296778) ? -0.0541525397770835 : ((f16 > 231.5) ? -0.0198829329735932 : 0.170973893627926)) : ((f9 > 9.436979) ? 0.14354736768267 : ((f32 > 0.06024682) ? 0.147246491784329 : 0.440775756888181))) : 0.000987550851735223) : -0.0449872501105444)))) : 0.532272344824875) : ((f18 > 7.860124) ? 0.352745096883182 : -0.132634024578179)) : ((f10 > 2.591825) ? ((f7 > -0.1680053) ? ((f10 > 5.616341) ? ((f29 > 4.5) ? ((f23 > 1.217493) ? ((f26 > 0.3794239) ? 0.104010771265932 : -0.192300190347137) : 0.175179695269152) : ((f32 > 0.06212721) ? ((f3 > 0.4977799) ? ((f0 > 1.455858) ? 0.100250384425843 : ((f13 > 17.07904) ? -0.191827426536384 : -0.0578784868872116)) : 0.027780575964225) : ((f25 > 0.4328707) ? ((f32 > 0.03216579) ? 0.0853755445384902 : ((f10 > 18.17398) ? ((f23 > 0.1365977) ? ((f25 > 0.8183883) ? 0.763939530954762 : ((f3 > 0.3145081) ? ((f6 > -0.07028989) ? 0.623812289319923 : 0.279518955188599) : 0.0137045944381472)) : ((f24 > 0.05574332) ? 0.12460364822713 : ((f3 > 0.3998359) ? 0.571609685817503 : 2.0631039868241))) : 0.16899877056711)) : 0.11733830587978))) : ((f19 > 1.843888) ? ((f12 > 1.245689) ? 0.226083963785754 : ((f12 > 0.02852431) ? 0.536840953252992 : 1.26272606623069)) : 0.210035374442873)) : ((f9 > 18.29306) ? -0.0703038977045474 : 0.0322883145648274)) : -0.123744414638407)) : ((f13 > 0.02541088) ? -0.0565724873550542 : 0.52422705845148));
            double treeOutput39 = ((f31 > 13.5) ? ((f16 > 220.5) ? -0.0459266764474824 : ((f32 > 0.2107053) ? ((f2 > 0.9058864) ? 0.123549011621123 : -0.202651029982447) : ((f5 > -1.303895E-06) ? ((f1 > 0.09247104) ? ((f25 > 0.378284) ? ((f18 > 2.572657) ? ((f27 > 0.4330106) ? ((f1 > 1.49885) ? 0.118107960932055 : -0.257330547687124) : 0.189197870045003) : ((f4 > 1.519449E-05) ? ((f5 > 0.008965526) ? ((f3 > 0.3833796) ? ((f27 > 0.2634143) ? -0.0800952291391101 : 0.0958842468039806) : ((f19 > 2.286003) ? -0.225940147701963 : -0.0153313753112632)) : ((f29 > 8.5) ? -0.251611504464529 : 0.17713522080938)) : -0.365513848449622)) : ((f2 > 0.03349049) ? 0.210192757527132 : -0.110121171132617)) : ((f23 > 0.3099543) ? -0.160721755395498 : 0.150967821732717)) : 0.645038101615179))) : ((f16 > 61.5) ? ((f27 > 0.08003467) ? ((f30 > 0.02317739) ? ((f32 > 0.3643579) ? 0.395435870455433 : ((f2 > 0.2096379) ? -0.090997688139302 : ((f26 > 0.1509575) ? 0.107076785928648 : -0.134212578085717))) : ((f20 > 1.527422) ? ((f29 > 1.5) ? ((f24 > 0.5355691) ? -0.00508655644399417 : ((f25 > 0.684077) ? ((f23 > 0.7405072) ? 0.154941687275836 : 0.457331282282919) : 0.132270496025265)) : -0.035325406532867) : -0.0636647825821349)) : ((f4 > 1.519449E-05) ? ((f27 > 0.02197022) ? ((f0 > 1.696755) ? ((f25 > 0.8183883) ? 0.124611939557253 : -0.356822688873308) : 0.00147212465406292) : ((f2 > 0.555065) ? ((f32 > 0.1207163) ? 0.40357297507498 : 0.190461262743478) : 0.137359141528025)) : ((f3 > 0.7317827) ? ((f29 > 8.5) ? 0.320881973107093 : ((f0 > 2.36744) ? 0.438067045296752 : ((f30 > 0.02880905) ? -0.194336110320301 : -0.105595471262644))) : -0.0663103612839313))) : 0.0269639875280183));
            double treeOutput40 = ((f15 > 0.7862563) ? ((f29 > 2.5) ? 0.064428732465334 : 0.0141739468679873) : ((f24 > 1.329) ? ((f28 > 0.6536086) ? -0.199925876782463 : 0.166453034538736) : ((f30 > 0.07080285) ? ((f4 > 0.7054959) ? -0.405697385229262 : ((f3 > 0.1969164) ? ((f32 > 0.1251566) ? ((f28 > 0.09136954) ? -0.0999580360258816 : 0.0927862594396098) : -0.156414864341839) : ((f1 > 1.428488) ? -0.183529343257406 : ((f6 > -0.1818074) ? 0.296916083909771 : 1.46481835546293)))) : ((f29 > 4.5) ? ((f18 > 1.65724) ? ((f16 > 202.5) ? ((f29 > 10.5) ? ((f25 > 0.684077) ? -0.119987407281132 : 0.195470212504954) : -0.139286823251007) : ((f28 > 0.1081162) ? 0.0113287440731332 : ((f25 > 0.5759066) ? ((f18 > 3.162284) ? 0.187983972170237 : 0.00625860685369594) : ((f23 > 1.287087) ? -0.0367629950007769 : ((f27 > 0.2015206) ? 0.0974286374645376 : ((f31 > 6.5) ? 0.386170960882929 : ((f3 > 0.4180971) ? ((f1 > 0.8333724) ? 0.89113902007453 : 0.329423319302389) : 0.00633505375681088))))))) : ((f22 > 0.2182757) ? -0.173266627737228 : -0.0069780060212382)) : ((f30 > 0.02920031) ? ((f16 > 61.5) ? -0.0915474490538881 : ((f29 > 2.5) ? 0.0237435719373496 : -0.083907389411986)) : ((f25 > 0.781347) ? ((f23 > 0.5138322) ? ((f0 > 2.208209) ? 0.389285870570293 : 0.00662251032383014) : ((f6 > 0.2542051) ? 0.324666450435538 : 0.151691481536431)) : ((f16 > 145.5) ? -0.152895000708206 : ((f29 > 2.5) ? ((f15 > 0.1920029) ? ((f18 > 1.158333) ? ((f15 > 0.4878115) ? 0.261285108579959 : ((f16 > 73.5) ? 0.0318041527372801 : 0.235222628197191)) : 0.0180274537812219) : -0.0422081056111745) : ((f23 > 0.4773526) ? -0.0594201475346533 : -0.00541207694416942)))))))));
            double treeOutput41 = ((f15 > 0.2845094) ? ((f31 > 12.5) ? ((f16 > 279.5) ? ((f31 > 38.5) ? 0.17012359594308 : -0.0886652595614141) : ((f32 > 0.05333724) ? ((f12 > 29.44574) ? -0.181993263951941 : ((f32 > 0.2305679) ? -0.11376984695107 : ((f13 > 15.23631) ? -0.0373555456122188 : ((f31 > 21.5) ? 0.192198264457681 : ((f1 > 0.8208281) ? 0.153327464914917 : ((f2 > 0.230726) ? ((f28 > -1.290886E-06) ? 0.0117442420244114 : 0.524044191946396) : 0.258371017557595)))))) : ((f14 > 0.3356808) ? 0.00339179589812815 : ((f2 > 0.0763697) ? 0.301950260351908 : 0.046814098765067)))) : ((f32 > 0.3643579) ? ((f0 > 1.064894) ? 0.0573611649878592 : 0.403796579653416) : ((f16 > 54.5) ? ((f32 > 0.05404603) ? ((f13 > 16.15499) ? ((f2 > 0.2585183) ? -0.0853418037713913 : 0.0906592542628076) : ((f32 > 0.1207163) ? ((f26 > 0.3691205) ? 0.0522403636648143 : ((f23 > 1.631086) ? -0.328731680587809 : ((f27 > 0.1150966) ? -0.0813788982814118 : 0.0626089864728246))) : ((f3 > 0.06121311) ? -0.0328329422382277 : 0.202717356801099))) : ((f9 > 2.816224) ? ((f31 > 7.5) ? ((f16 > 119.5) ? -0.0231931094911749 : ((f18 > 1.022128) ? ((f6 > -0.02993276) ? ((f32 > 0.04167926) ? 0.196219760884487 : ((f15 > 0.5441098) ? 0.947094839008117 : 0.490096436265721)) : 0.128001716016151) : 0.121734425142897)) : ((f10 > 17.93593) ? ((f7 > -0.1750269) ? ((f25 > 0.847397) ? ((f32 > 0.0178644) ? 0.227399732716671 : 0.911540420887223) : 0.164902565553137) : -0.0232389064389141) : ((f7 > 0.999862) ? ((f25 > 0.8325248) ? 0.748524651988313 : 0.281132433725527) : -0.0800720873174343))) : 0.18134787107459)) : 0.0467345278886445))) : ((f30 > 0.04001511) ? -0.122663487320253 : -0.0535363150251951));
            double treeOutput42 = ((f11 > 7.912674) ? ((f5 > 0.003619287) ? 0.123022474059046 : 0.024275504543301) : ((f18 > 6.075307) ? 0.125415737511822 : ((f30 > 0.09095263) ? ((f26 > 0.05588391) ? ((f21 > 2.343049) ? -0.389836885494732 : -0.0872197068808303) : ((f1 > 1.071631) ? 0.0258119117270551 : 1.08082679092798)) : ((f15 > 0.1920029) ? ((f29 > 3.5) ? ((f23 > 0.1560663) ? ((f16 > 193.5) ? ((f29 > 9.5) ? ((f10 > 18.23078) ? 0.364892663727403 : 0.0327949713525914) : ((f30 > 0.04080816) ? -0.279584510572259 : -0.0624105101527626)) : ((f28 > 0.04338297) ? ((f31 > 17.5) ? 0.167803888310574 : ((f18 > 2.167583) ? ((f2 > 0.6819611) ? -0.044415184852789 : 0.114948509376093) : -0.0854377785367742)) : ((f19 > 3.05059) ? ((f0 > 1.626267) ? 0.212102994712172 : ((f4 > 1.519449E-05) ? 0.0338876637342558 : -0.154901438624347)) : ((f18 > 1.676616) ? ((f23 > 1.631086) ? -0.280893235186092 : ((f27 > 0.2466612) ? ((f17 > 3.469717) ? 0.226194766669574 : -0.204917268901973) : ((f18 > 3.636645) ? ((f27 > 0.009942396) ? 0.264091975129416 : 0.71077460136447) : ((f30 > 0.06450012) ? 0.0451336826375457 : ((f32 > 0.09432109) ? 0.306276134009493 : 0.186424880460093))))) : 0.0607605782687326)))) : ((f26 > 0.1225576) ? -0.234859292037659 : 0.532342741374599)) : ((f30 > 0.0217307) ? ((f16 > 69.5) ? -0.0854689835926484 : -0.0152740057243004) : ((f29 > 1.5) ? ((f21 > 0.7866079) ? ((f32 > 0.08649219) ? 0.234247775226917 : 0.135638165269275) : ((f16 > 156.5) ? -0.111738611255938 : 0.0746384773462009)) : ((f30 > 0.01471059) ? ((f0 > 5.913356E-05) ? ((f32 > 0.1332605) ? -0.206742062003523 : -0.129745783651243) : 0.264415273612296) : -0.00116217049895749)))) : -0.105367958585166))));
            double treeOutput43 = ((f0 > 0.1231926) ? ((f20 > 0.391529) ? ((f18 > 2.016162) ? ((f25 > 0.3869354) ? ((f29 > 1.5) ? ((f30 > 0.04764608) ? ((f24 > 1.228056) ? 0.0981177555631293 : ((f27 > 0.1996688) ? ((f12 > 8.229173) ? ((f23 > 0.4514873) ? -0.186878183499099 : 0.271000572984804) : -0.0456613467971984) : ((f31 > 2.5) ? ((f18 > 3.414886) ? 0.0908540170070732 : -0.0214079187046776) : -0.266033852787742))) : ((f11 > 2.29283) ? 0.139601965555575 : 0.0318470731527654)) : -0.0968730010459466) : ((f5 > 0.1166317) ? ((f25 > 0.2318654) ? ((f26 > 0.3465821) ? ((f4 > 0.2239079) ? 0.290484351694038 : 0.708125317657783) : ((f3 > 0.2196331) ? -0.188435954704536 : 0.328140356448774)) : 0.187260480223186) : ((f2 > 0.07084043) ? 0.337540847141629 : 0.102229170835574))) : ((f28 > 0.2464993) ? ((f28 > 0.4203187) ? 0.874899983486024 : ((f14 > 0.9958677) ? 1.27032906797132 : 0.182153819200529)) : ((f30 > 0.03446827) ? ((f6 > -0.06400484) ? -0.0942538191940448 : -0.0139043773195039) : ((f26 > 0.5607868) ? ((f0 > 1.858214) ? 0.586114615643665 : ((f27 > 0.001190391) ? ((f26 > 0.795416) ? 0.596980786768939 : 0.180353515877789) : -0.062929919058567)) : ((f18 > 0.8627284) ? 0.00684552776122191 : ((f30 > 0.01249541) ? -0.103377663150483 : ((f23 > 0.1965906) ? -0.0573535174200782 : 0.166092472158565))))))) : ((f1 > 0.00994114) ? ((f23 > 0.4773526) ? 0.0142379941481177 : ((f32 > 0.0137912) ? ((f21 > 0.09460497) ? 0.137630389323135 : 0.406752551115249) : 0.21303281426374)) : 0.0948210103656637)) : ((f28 > 0.02591503) ? ((f25 > 0.03439728) ? -0.239188080302399 : 3.27354661022204) : ((f31 > 4.5) ? ((f0 > 5.913356E-05) ? 0.0115787912192404 : 0.268983971769442) : -0.179202390513771)));
            double treeOutput44 = ((f15 > 0.6547681) ? ((f29 > 2.5) ? ((f30 > 0.04614702) ? ((f1 > 0.7230155) ? ((f30 > 0.07080285) ? -0.00600493620430629 : 0.0919180526005895) : -0.0483667761290106) : ((f1 > 0.358116) ? ((f16 > 92.5) ? 0.0698252080539487 : ((f30 > 0.03487648) ? ((f16 > 60.5) ? 0.095087313150533 : 0.288051177879642) : 0.322779550361352)) : 0.00256024143526246)) : ((f30 > 0.01428005) ? ((f10 > 18.29356) ? ((f25 > 0.7929175) ? 0.0493309337430553 : ((f20 > 1.670022) ? ((f32 > 0.1207163) ? ((f28 > 0.165318) ? -0.0383253475072447 : -0.216112708713929) : -0.0644881560366513) : ((f6 > -0.04282933) ? ((f28 > 0.119511) ? 0.0952243218547221 : ((f23 > 0.2011348) ? -0.134160839503311 : 0.0073278922012088)) : 0.00409425963977557))) : 0.00784912611437747) : ((f2 > 0.4955539) ? ((f16 > 86.5) ? 0.0263917526822381 : ((f23 > 0.1774827) ? ((f25 > 1.043569) ? 0.560331671639174 : ((f6 > 0.06035792) ? 0.169794575360958 : 0.0560267311146198)) : ((f32 > 0.009853834) ? 0.396817263748832 : 3.30646090421264))) : 0.0250292345943327))) : ((f4 > 0.0949056) ? ((f11 > 6.045397) ? ((f12 > 33.84128) ? ((f18 > 1.237303) ? 0.31873500890754 : 0.0723762906608457) : -0.00706396208052917) : ((f13 > 9.125596) ? ((f20 > 2.802831) ? -0.20659351986478 : -0.0705720926284298) : ((f17 > 4.464258) ? 0.044376555250795 : -0.04015340528956))) : ((f4 > 1.519449E-05) ? ((f27 > 0.02556339) ? ((f31 > 9.5) ? ((f1 > 0.06260691) ? ((f16 > 139.5) ? -0.0120652291780322 : 0.0990561122189815) : -0.128270023018691) : -0.0237339681027813) : ((f2 > 0.555065) ? ((f31 > 20.5) ? -0.329573923150674 : ((f32 > 0.1207163) ? 0.32694125153722 : 0.149618664025479)) : 0.108800499559629)) : -0.0623471209606512)));
            double treeOutput45 = ((f16 > 75.5) ? ((f7 > 1) ? 0.172243090850333 : ((f14 > 0.3421603) ? ((f13 > 2.060376) ? ((f25 > 0.6395183) ? -0.248961650952591 : -0.185920384968006) : ((f10 > 13.8726) ? 0.232891912387946 : -0.0483465029391648)) : ((f31 > 22.5) ? ((f27 > 0.02197022) ? 0.111638304179182 : -0.277831991019556) : ((f30 > 0.04128736) ? ((f18 > 3.803536) ? ((f30 > 0.1270167) ? ((f26 > 0.3405882) ? -0.0749123310891499 : -0.389165181981698) : ((f3 > 0.7119519) ? -0.0833177045661947 : 0.154477171923567)) : ((f32 > 0.1626157) ? ((f1 > 0.334854) ? 0.0269576932425469 : 0.516906045030333) : ((f6 > 0.2687221) ? -0.173916697811382 : -0.0790348386075846))) : ((f25 > 0.9742545) ? ((f6 > 0.1586699) ? ((f30 > 0.01785051) ? 0.0663731347671324 : 0.34172364744221) : 0.0194914938057587) : ((f26 > 0.795416) ? -0.23667681990634 : ((f32 > 0.1075094) ? ((f29 > 1.5) ? ((f31 > 2.5) ? ((f31 > 3.5) ? 0.0986638664135252 : -0.140890007824205) : 0.650813062475781) : -0.0593613802335608) : ((f18 > 4.236958) ? 0.206035003597509 : -0.0250756233915327)))))))) : ((f31 > 8.5) ? ((f10 > 1.686399) ? ((f14 > 0.04210993) ? ((f9 > 13.25933) ? -0.0565933982399643 : ((f15 > 0.5019789) ? ((f7 > 0.4097912) ? ((f12 > 3.951534) ? 0.421457198333927 : -0.101517605131663) : ((f18 > 0.7224665) ? 0.389690019522838 : 0.065649691875336)) : ((f11 > 1.360109) ? 0.302753030544998 : 0.07847246719998))) : ((f13 > 0.9012095) ? ((f23 > 0.258561) ? ((f29 > 2.5) ? 0.0574883300950921 : ((f32 > 0.1430221) ? ((f28 > 0.03427891) ? -0.068064890923633 : -0.243126167228349) : 0.00796281757190588)) : 0.151509284108813) : -0.338404148005107)) : -0.0472596395334356) : 0.00870008589386177));
            double treeOutput46 = ((f6 > -0.1935124) ? ((f31 > 21.5) ? ((f16 > 600.5) ? -0.147451111092812 : ((f27 > 0.01867885) ? 0.100053028217335 : -0.253962321391181)) : ((f16 > 207.5) ? ((f29 > 13.5) ? 0.147567731312897 : ((f24 > 0.7823775) ? -0.171169147716954 : -0.0960005296912732)) : ((f32 > 0.008768726) ? ((f32 > 0.05102542) ? ((f13 > 17.07904) ? -0.0571445521272933 : ((f12 > 17.16026) ? ((f14 > 0.03883861) ? 0.0457070456965849 : -0.12829631863266) : ((f13 > 0.8539063) ? ((f13 > 1.065469) ? ((f12 > 1.928594) ? ((f12 > 2.061996) ? ((f13 > 1.922309) ? ((f31 > 11.5) ? ((f6 > 0.2736513) ? 0.0050099817274613 : 0.0872102374612392) : 0.00724510246276527) : ((f14 > 0.1943352) ? 0.000805228391352515 : -0.204811765110354)) : 0.154248150322431) : ((f14 > 0.3801314) ? ((f7 > 0.1979205) ? -0.192838306808011 : 0.130015634245258) : -0.109041842594319)) : ((f12 > 1.128524) ? -0.0240941651620093 : ((f7 > 0.4202187) ? 0.0673227912616071 : ((f7 > -0.1666666) ? 0.567498507807257 : 0.167041434337026)))) : ((f14 > 0.2240143) ? ((f26 > 0.5894637) ? 0.241076973085527 : -0.0554467457409102) : -0.267498572485162)))) : ((f6 > 0.3669006) ? ((f25 > 0.6446771) ? ((f31 > 2.5) ? ((f14 > 0.1550481) ? 0.0658248355067648 : 0.286525023008816) : -0.0906070892559172) : ((f6 > 0.918069) ? 0.360212004293855 : 0.0873899639970187)) : ((f10 > 18.14574) ? ((f7 > -0.1723428) ? ((f30 > 0.008768586) ? ((f25 > 0.6904409) ? 0.220267309422275 : 0.0379103022285307) : 0.422522670845466) : 0.0551972164125747) : ((f7 > 0.9878271) ? 0.251830854070329 : ((f31 > 8.5) ? ((f15 > 0.449443) ? 0.205991487329881 : 0.0324073269358666) : -0.0211425259000957))))) : -0.131733673775636))) : -0.15794783061848);
            double treeOutput47 = ((f15 > 0.4671022) ? ((f29 > 3.5) ? ((f30 > 0.06058567) ? ((f18 > 3.106588) ? ((f30 > 0.1378544) ? -0.0907990770789988 : 0.0696225598504959) : -0.0597199512289691) : ((f16 > 464.5) ? -0.0924405542971928 : ((f18 > 1.269553) ? ((f23 > 0.1812081) ? 0.112657824083428 : -0.129215171132247) : ((f3 > 0.17889) ? -0.0556360936311486 : 0.20568418970072)))) : ((f30 > 0.04001511) ? -0.0522468104922617 : ((f29 > 1.5) ? ((f18 > 1.544705) ? ((f16 > 73.5) ? ((f30 > 0.0200087) ? -0.0434122584802443 : ((f16 > 149.5) ? -0.0406853257144171 : 0.244995684320654)) : ((f30 > 0.02683053) ? ((f29 > 2.5) ? 0.241197313886936 : 0.0352922145229933) : ((f32 > 0.179564) ? ((f2 > 0.8188744) ? 0.142267146706486 : -0.251186402914979) : ((f15 > 0.7448221) ? 0.572970427040247 : ((f30 > 0.0217307) ? 0.200607212643701 : 0.556055091139178))))) : ((f30 > 0.02423923) ? -0.0512600867660582 : ((f2 > 0.6577429) ? 0.152034513068741 : ((f26 > 0.1056351) ? -0.0158793615311178 : 0.18867726112872)))) : ((f30 > 0.01550034) ? ((f23 > 0.1003069) ? ((f31 > 8.5) ? -0.187721965803552 : -0.128008261438856) : ((f2 > 0.6757982) ? ((f21 > 0.599337) ? -0.215121190850074 : 0.418380881781232) : -0.0317103052736951)) : ((f16 > 67.5) ? ((f30 > 0.01041215) ? -0.112083036937776 : 0.000476538015795188) : ((f24 > 0.01760866) ? ((f30 > 0.01123201) ? ((f2 > 0.9058864) ? 0.287701122506599 : 0.0201057771055935) : 0.198431146374794) : 0.0627155549662958)))))) : ((f14 > 0.1322404) ? ((f25 > 0.5088348) ? -0.110790336864412 : -0.0634348203867444) : ((f30 > 0.04201297) ? -0.0610456363011135 : ((f29 > 3.5) ? ((f0 > 1.477073) ? 0.200492976818233 : 0.0389892339055638) : -0.0159650511936716))));
            double treeOutput48 = ((f16 > 55.5) ? ((f9 > 4.778819) ? ((f16 > 114.5) ? ((f19 > 1.205926) ? ((f32 > 0.09432109) ? ((f3 > 0.4105584) ? ((f3 > 0.7317827) ? -0.0596643048545917 : ((f31 > 3.5) ? 0.114026710457687 : -0.197776337601038)) : -0.0826651956942386) : ((f32 > 0.0468849) ? ((f6 > 0.2786338) ? -0.183189869842379 : -0.102997440466854) : -0.0957219503456363)) : -0.0632813448750492) : ((f29 > 4.5) ? ((f27 > 0.147201) ? -0.00466423542826388 : 0.0864443828240418) : ((f30 > 0.03772757) ? ((f32 > 0.1363017) ? 0.0340218467761495 : -0.0739183951993668) : ((f29 > 2.5) ? ((f15 > 0.6665833) ? ((f1 > 0.3076679) ? 0.167768293478758 : -0.00852822107340159) : 0.0313012678259737) : ((f30 > 0.0212679) ? -0.0625357587923788 : ((f29 > 1.5) ? ((f27 > 0.2149771) ? 0.174729203263289 : 0.0432197944789126) : ((f30 > 0.01110671) ? -0.0881838432781355 : ((f25 > 0.4369328) ? 0.0773744720404708 : -0.0384393817217477)))))))) : ((f31 > 32.5) ? 0.192335677840347 : ((f28 > 0.03961268) ? ((f1 > 1.144978) ? ((f2 > 1.819301) ? -0.519424115502964 : 0.144034223549064) : ((f2 > 0.1026826) ? ((f10 > 1.635757) ? ((f7 > -0.1705405) ? -0.00564286762224306 : ((f14 > 0.07767723) ? -0.0212177515693173 : ((f25 > 0.595188) ? 0.0101470111988757 : -0.226127070157459))) : -0.208501791427246) : 0.0856971934906894)) : ((f5 > 4.309524E-05) ? ((f29 > 4.5) ? ((f2 > 0.1026826) ? 0.245856426913231 : -0.131838067911238) : 0.0648102818980677) : 0.0183703539348789)))) : ((f31 > 4.5) ? ((f32 > 0.05562594) ? ((f0 > 0.5520531) ? ((f18 > 3.05299) ? 0.0741364460528632 : -0.0159851447884289) : 0.0802391511932572) : ((f6 > 0.183035) ? 0.212599375851221 : 0.105892693503084)) : 0.0170347940919423));
            double treeOutput49 = ((f15 > 0.8064615) ? ((f1 > 0.5470313) ? ((f13 > 0.9012095) ? 0.0632650698508061 : ((f10 > 18.4178) ? -0.302522710947776 : -0.0151050534980184)) : 0.0196382525801516) : ((f15 > 0.1038589) ? ((f24 > 1.733555) ? 0.204633769946134 : ((f30 > 0.1022561) ? ((f27 > 0.7585866) ? -0.479971377013892 : ((f1 > 0.4171919) ? ((f25 > 1.450612) ? -0.237032921238213 : ((f26 > 0.1797282) ? ((f1 > 1.49885) ? 0.067995025312878 : -0.143951109548483) : 0.441905277469316)) : 0.777420692946989)) : ((f1 > 1.094739) ? ((f26 > 0.7179694) ? ((f2 > 1.09351) ? ((f22 > 1.588133) ? -0.389751574620691 : 0.107380024143191) : ((f1 > 1.691513) ? 0.186369306970552 : -0.206400678741155)) : 0.0851121500901702) : ((f30 > 0.04201297) ? ((f32 > 0.1578004) ? ((f3 > 0.9333973) ? ((f2 > 1.122298) ? 0.363063046557605 : -0.395331305453612) : ((f5 > 0.1454616) ? -0.0388214855596088 : 0.179413605682426)) : ((f16 > 82.5) ? -0.137575972933767 : -0.0362395190337574)) : ((f28 > 0.3247706) ? 0.319636019310846 : ((f25 > 1.158015) ? 0.173759847115264 : ((f13 > 0.02541088) ? ((f14 > 0.2113238) ? -0.0689566894486708 : ((f29 > 2.5) ? ((f0 > 1.434483) ? ((f32 > 0.08326913) ? 0.256357679921391 : 0.0756726682665262) : ((f23 > 1.217493) ? -0.186245421019851 : ((f31 > 19.5) ? ((f2 > 0.08278456) ? ((f32 > 0.05102542) ? 0.0898730658992063 : 0.321389006438553) : -0.196734532187637) : ((f18 > 1.735561) ? ((f2 > 0.6881586) ? -0.022468347790801 : ((f27 > 0.1150966) ? 0.0518400945178998 : 0.231311150270214)) : 0.00260123383564859)))) : ((f30 > 0.01723713) ? -0.0489821717631128 : 0.00925531602260679))) : 0.304757468890972))))))) : ((f30 > 0.02128645) ? -0.238533091042212 : -0.129986486037359)));
            double treeOutput50 = ((f7 > -0.3058829) ? ((f7 > -0.05732074) ? ((f12 > 3.82842) ? ((f10 > 8.795653) ? -0.0783751477808844 : ((f6 > 0.221998) ? ((f10 > 2.053907) ? ((f2 > 0.05656329) ? ((f16 > 73.5) ? 0.0923867618907602 : 0.216767631182805) : 0.00808049430264046) : 0.0258442808493563) : 0.00238432232580173)) : ((f4 > 0.4593562) ? 0.170766450460191 : -0.123668393800467)) : ((f10 > 2.941202) ? ((f9 > 18.41715) ? ((f12 > 0.6421817) ? ((f6 > -0.05015822) ? 0.141610623394534 : -0.0243128474514679) : ((f32 > 0.1528009) ? -0.411620773538369 : -0.181924694232719)) : ((f7 > -0.1740668) ? ((f10 > 5.7434) ? ((f10 > 16.75527) ? ((f14 > 0.3538899) ? 0.280566865397956 : ((f30 > 0.01162228) ? ((f29 > 1.5) ? ((f6 > -0.09463143) ? ((f30 > 0.01887332) ? ((f29 > 5.5) ? 0.123046535795117 : ((f5 > 0.1535625) ? 0.0967555461495355 : ((f7 > -0.1227722) ? 0.328054421717772 : -0.0351363506670631))) : 0.126653756888634) : 0.207315353624027) : -0.0833358791243648) : ((f31 > 0.5) ? ((f32 > 0.03436223) ? ((f31 > 6.5) ? 0.169023739308837 : -0.00604596760296126) : ((f26 > 0.5607868) ? 0.792069317445439 : 0.429245574364033)) : 0.319554564429567))) : -0.0163062452749472) : ((f2 > 0.5437956) ? ((f12 > 1.036453) ? 0.219748601781371 : 0.517698856323641) : 0.235103391706631)) : ((f24 > 0.01760866) ? ((f23 > 0.3549647) ? ((f14 > 0.04040816) ? 0.113684848416556 : ((f6 > -0.1001967) ? ((f25 > 0.2599556) ? ((f2 > 0.510906) ? ((f32 > 0.05562594) ? -0.0494415068678209 : 0.0633868507327938) : -0.147181690784127) : 0.0805007954260306) : 0.132116218394934)) : ((f2 > 0.112146) ? 0.0914525341389611 : 0.503416964540083)) : -0.0502162526907525))) : -0.0786374944722605)) : -0.0469596427117591);
            double treeOutput51 = ((f16 > 87.5) ? -0.0260813262405373 : ((f31 > 6.5) ? ((f32 > 0.0545509) ? ((f12 > 16.1921) ? ((f25 > 0.2142025) ? -0.0520967289618485 : 0.161878895154671) : ((f23 > 0.5033655) ? ((f0 > 1.054605) ? ((f27 > 0.05912145) ? ((f28 > 0.03961268) ? ((f3 > 0.5897081) ? ((f23 > 0.8633971) ? ((f0 > 1.415576) ? 0.273456518394155 : -0.037812867765953) : 0.587725967501406) : ((f2 > 0.7004806) ? -0.0879825265836266 : ((f3 > 0.4033637) ? 0.202412602114971 : -0.0153122123576192))) : 0.172187061216023) : -0.0380509086936997) : ((f21 > 1.315003) ? -0.228457108356832 : ((f1 > 0.7634702) ? ((f2 > 0.6518896) ? 0.0142310367781561 : 0.292684246926972) : ((f25 > 0.2812213) ? -0.0636078463062824 : 0.0655765810624021)))) : ((f0 > 1.086788) ? ((f26 > 0.4785053) ? 0.227682653546733 : -0.197585442591605) : ((f25 > 0.4530378) ? ((f4 > 0.1272882) ? 0.0809364795851197 : ((f27 > 0.002698111) ? 0.222156933634106 : -0.106048426301892)) : ((f18 > 1.563098) ? 0.233118121321901 : 0.000468873719204157))))) : ((f15 > 0.5926044) ? ((f18 > 1.127722) ? 0.444921693107672 : ((f2 > 0.4628267) ? 0.405365057987919 : 0.047652113715874)) : ((f6 > 0.04710531) ? ((f1 > 0.4120788) ? 0.174167044775943 : 0.0701173232440839) : 0.000941680651513304))) : ((f5 > -0.0002845971) ? ((f5 > 0.05829879) ? -0.0806730475113539 : ((f29 > 2.5) ? ((f30 > 0.06450012) ? ((f6 > -0.1234417) ? -0.071175903270284 : 0.197614734103219) : ((f15 > 0.3036077) ? 0.129031383986137 : -0.00043935798157453)) : ((f28 > 0.002456723) ? ((f3 > 0.5994364) ? 0.525963242517191 : ((f25 > 0.0604436) ? 0.0182100280034256 : ((f1 > 0.6444818) ? 0.942614143043011 : 0.407959974883439))) : -0.0128261126209109))) : -0.276865357274593)));
            double treeOutput52 = ((f31 > 14.5) ? ((f13 > 15.23631) ? -0.0377728563022177 : ((f10 > 2.092505) ? 0.0722971258398261 : -0.00300620975602593)) : ((f27 > 0.08449583) ? ((f13 > 8.201846) ? ((f30 > 0.03772757) ? ((f23 > 0.3657564) ? -0.093394175152232 : ((f1 > 0.7427227) ? 0.255719207023848 : -0.034839853178536)) : ((f25 > 0.7509996) ? 0.0726460823035068 : ((f28 > 4.172743E-05) ? -0.00115757034152176 : -0.0854260783746464))) : ((f27 > 0.3349689) ? ((f0 > 1.396596) ? ((f23 > 1.287087) ? ((f26 > 1.080976) ? -0.290099952759306 : 0.143373120429211) : 0.365776958962703) : 0.00678116450326135) : -0.0125300328449173)) : ((f4 > 1.519449E-05) ? ((f27 > 0.01600741) ? ((f17 > 5.799238) ? ((f25 > 0.781347) ? ((f26 > 0.4258785) ? 0.27918798635151 : -0.107201591579078) : -0.296555545923598) : ((f18 > 1.221413) ? ((f5 > -1.303895E-06) ? ((f23 > 0.5138322) ? 0.0182150122971831 : ((f9 > 9.571646) ? 0.0478033584249437 : ((f26 > 0.3238384) ? 0.00513354392424845 : ((f0 > 1.109384) ? -0.11304457119381 : ((f10 > 2.513241) ? 0.33977669420593 : 0.181124653378097))))) : 0.70434200443996) : ((f3 > 0.1682506) ? ((f23 > 0.2983109) ? -0.0983401527112993 : ((f25 > 0.4490376) ? 0.257868548083461 : 0.017441168761595)) : ((f2 > 0.8355188) ? 1.15760441346547 : 0.0458654193126281)))) : ((f19 > 1.843888) ? 0.196389401210911 : ((f28 > 0.0191633) ? 0.780489068144667 : 0.133295239066276))) : ((f19 > 2.620512) ? -0.0454582669601536 : ((f31 > 3.5) ? ((f1 > 0.8739062) ? ((f23 > 1.175754) ? 0.0960022866484987 : ((f32 > 0.1052382) ? 0.874096605504231 : ((f2 > 0.6350518) ? 0.0163342700951441 : 0.663632962979507))) : 0.0725775532584171) : ((f20 > 1.558039) ? -0.231919181815544 : -0.0639982393148494))))));
            double treeOutput53 = ((f15 > 0.7448221) ? ((f31 > 8.5) ? ((f12 > 28.72018) ? -0.107087575000725 : ((f28 > -1.290886E-06) ? 0.0447453424974601 : 0.353499770459887)) : 0.0134154819599988) : ((f16 > 63.5) ? ((f30 > 0.0217307) ? ((f29 > 5.5) ? ((f30 > 0.08476043) ? ((f1 > 1.275451) ? ((f30 > 0.1270167) ? -0.108875238671953 : 0.113001608452227) : -0.199609756602197) : ((f18 > 2.114986) ? ((f26 > 0.2049616) ? ((f15 > 0.1615864) ? ((f16 > 146.5) ? ((f29 > 8.5) ? ((f10 > 13.63814) ? 0.265289272373486 : 0.0336658821723739) : -0.122231781249855) : ((f30 > 0.04445453) ? ((f1 > 0.9190375) ? 0.181822815796962 : -0.0338634591821554) : 0.253639752702727)) : -0.113186895834617) : 0.341277164990792) : ((f28 > 0.07247432) ? -0.188622135104675 : ((f32 > 0.1061548) ? 0.206629503323656 : ((f2 > 0.6696985) ? -0.224762712471512 : 0.0298625799656957))))) : ((f20 > 2.506219) ? ((f25 > 0.9742545) ? ((f27 > 0.5500337) ? -0.2126864115858 : 0.088589612206356) : ((f0 > 2.36744) ? 0.601014139415964 : -0.203084872455327)) : ((f32 > 0.1626157) ? 0.0959236300298373 : -0.0644060596287224))) : ((f25 > 0.6499294) ? ((f14 > 0.2063895) ? -0.121870975502147 : ((f6 > 0.2542051) ? ((f31 > 4.5) ? ((f32 > 0.03587122) ? 0.122189905247466 : ((f6 > 0.622574) ? 0.619598062162491 : 0.435779935948309)) : 0.0447183615462734) : 0.0345958317827127)) : ((f3 > 0.2433768) ? ((f0 > 0.8858363) ? ((f19 > 1.072145) ? ((f22 > 0.2972097) ? -0.149787536157175 : 0.0286939178065932) : ((f5 > 0.003619287) ? 0.344894815285678 : 0.00822625219539558)) : -0.0925878127874744) : ((f29 > 2.5) ? ((f10 > 2.89904) ? 0.18829462363259 : 0.00518422856135487) : -0.0503961258056627)))) : 0.0119675857158873));
            double treeOutput54 = ((f15 > 0.4210348) ? ((f31 > 10.5) ? ((f12 > 24.09407) ? ((f25 > 0.4609665) ? -0.157894083168478 : -0.009103339239184) : ((f28 > -1.290886E-06) ? ((f14 > 0.539072) ? -0.0314216205710104 : ((f14 > 0.04494949) ? ((f24 > 0.1009018) ? ((f10 > 1.490089) ? ((f16 > 161.5) ? 0.0831564213526494 : ((f9 > 14.28095) ? -0.038906337495183 : ((f30 > 0.02447358) ? ((f23 > 0.3211957) ? 0.212439835405267 : -0.0886765957740936) : 0.307745756353225))) : -0.151286117960784) : -0.0982050338355544) : ((f23 > 0.3549647) ? ((f29 > 1.5) ? ((f30 > 0.01959911) ? ((f6 > -0.06875998) ? ((f1 > 0.677354) ? 0.0238180036328628 : ((f4 > 0.1527046) ? -0.0069345448933958 : -0.11397875931601)) : ((f9 > 17.15072) ? 0.015310945855745 : 0.152417985868412)) : ((f3 > 0.4690144) ? 0.273109501718207 : ((f23 > 1.175754) ? -0.267994713638883 : 0.12139014357828))) : ((f18 > 1.142924) ? -0.242388562575867 : -0.00380492723337965)) : 0.107349334746284))) : 0.414653695983628)) : ((f5 > -0.0002845971) ? ((f12 > 1.875341) ? ((f16 > 60.5) ? ((f23 > 1.631086) ? ((f0 > 2.008788) ? -0.00284520585255292 : -0.290754074511129) : ((f14 > 0.4132458) ? -0.165681667786438 : ((f25 > 1.092159) ? 0.111419173709235 : ((f29 > 7.5) ? ((f26 > 0.3559913) ? ((f27 > 0.02338771) ? 0.0373587352409429 : ((f25 > 0.7509996) ? 0.0686050876580739 : 0.557038115781122)) : -0.00919862719255261) : ((f30 > 0.03138225) ? -0.0337890002558714 : ((f5 > 4.309524E-05) ? 0.0656060441104331 : -0.000995681479716527)))))) : 0.0385628962352837) : ((f13 > 2.1132) ? -0.160289328099179 : ((f26 > 0.5266525) ? 0.182277130606202 : -0.0119946839590857))) : ((f2 > 0.5723559) ? 0.594715989031755 : -0.191820837022975))) : -0.029683617325809);
            double treeOutput55 = ((f16 > 101.5) ? ((f10 > 0.01398623) ? ((f13 > 0.02541088) ? ((f14 > 0.2240143) ? ((f25 > 0.6096481) ? ((f10 > 13.15763) ? 0.163318564709191 : -0.164346863665561) : -0.0915171421997371) : ((f30 > 0.04614702) ? ((f1 > 0.2384984) ? ((f24 > 1.091105) ? 0.0300033928214169 : -0.116495223601669) : 0.568508523550241) : ((f29 > 8.5) ? 0.12347494337974 : ((f25 > 1.092159) ? 0.149569714050973 : -0.0152931914889094)))) : 0.298639470853558) : 0.196288178679775) : ((f31 > 11.5) ? ((f32 > 0.2107053) ? ((f2 > 0.9449936) ? 0.141541608023153 : -0.137957013813912) : ((f18 > 2.467326) ? ((f27 > 0.4001538) ? -0.094697243560144 : 0.173850464762916) : ((f2 > 0.2377282) ? ((f5 > -1.303895E-06) ? ((f23 > 0.4773526) ? ((f27 > 0.1348027) ? ((f27 > 0.2234133) ? -0.0616201289669033 : 0.0792907664500139) : ((f29 > 3.5) ? ((f27 > 0.02059509) ? 0.0949994588476832 : -0.216793681073792) : ((f6 > -0.04421057) ? ((f32 > 0.06328329) ? -0.217780724176348 : -0.0345173711843403) : -0.00151097648234576))) : 0.0495776978360539) : 0.517689832490558) : ((f18 > 0.8064011) ? ((f15 > 0.1525831) ? ((f30 > 0.0212679) ? 0.236349760084032 : 0.606070747878678) : 0.110733657028356) : 0.0663607911643803)))) : ((f24 > 1.733555) ? 0.295690394386973 : ((f30 > 0.05266714) ? ((f6 > -0.1249759) ? -0.05077380683702 : ((f21 > 0.1214078) ? -0.0084031971086512 : 0.265803889397403)) : ((f27 > 0.4533951) ? 0.168198926150375 : ((f6 > -0.1522937) ? ((f18 > 2.903062) ? ((f25 > 0.5565051) ? 0.0138105492449002 : ((f28 > 0.1791834) ? -0.0371439104941841 : ((f15 > 0.1038589) ? 0.289187788471325 : -0.200799726350925))) : ((f5 > 0.2931625) ? 0.178304425832946 : 0.00933131295318178)) : -0.122695933718495))))));
            double treeOutput56 = ((f31 > 26.5) ? ((f28 > 0.002456723) ? ((f1 > 0.1963926) ? ((f23 > 0.2983109) ? ((f2 > 0.06387354) ? 0.215911941593744 : -0.185951064761322) : -0.235347604690227) : ((f14 > 0.1259921) ? -0.252928583195874 : 0.150120641149176)) : ((f29 > 13.5) ? -0.376289247021025 : 0.242282013627186)) : ((f16 > 279.5) ? -0.105603635707169 : ((f15 > 0.1480016) ? ((f29 > 3.5) ? ((f30 > 0.06450012) ? -0.0178234341215865 : ((f18 > 2.501363) ? 0.107882615208716 : ((f27 > 0.120344) ? ((f0 > 0.959049) ? 0.0594937858659912 : -0.0771817695991286) : ((f2 > 0.8033784) ? ((f30 > 0.0254747) ? -0.117723899609286 : 0.171025527993643) : ((f1 > 0.4817365) ? ((f32 > 0.1332605) ? 0.35265255803094 : ((f26 > 0.3465821) ? 0.018835317872233 : 0.196794601765496)) : ((f26 > 0.2310707) ? -0.0468278394560085 : ((f28 > 0.0493812) ? -0.0618153519362803 : 0.169371224467304))))))) : ((f30 > 0.03060765) ? -0.0389293521993024 : ((f29 > 1.5) ? ((f16 > 57.5) ? ((f30 > 0.01961497) ? -0.0220355457950158 : ((f10 > 18.17398) ? ((f25 > 0.684077) ? 0.219388692375674 : 0.0777486265052834) : 0.0478229816544932)) : ((f1 > 0.1922342) ? ((f15 > 0.6622483) ? ((f24 > 0.3138551) ? ((f12 > 0.9221007) ? 0.561096902879683 : -0.0397454352503181) : 0.179723042385213) : 0.0972003182692407) : -0.0627964459022896)) : ((f30 > 0.01315249) ? ((f0 > 5.913356E-05) ? -0.089570079044133 : ((f27 > 0.1313349) ? -0.240831906296865 : 0.272405759448522)) : ((f0 > 0.09902693) ? ((f3 > 0.0761422) ? -0.000441915311419809 : ((f24 > 0.01213391) ? ((f24 > 0.1009018) ? 0.00605966121713261 : ((f9 > 3.982422) ? 0.345754774766207 : 0.584047854018183)) : 0.179141044785375)) : -0.11436813477018))))) : -0.096541866989869)));
            double treeOutput57 = ((f32 > 0.3643579) ? ((f31 > 2.5) ? 0.114562472746339 : 0.938091395082106) : ((f11 > 2.936746) ? ((f5 > 0.00263009) ? ((f12 > 40.48455) ? 0.302335475336128 : 0.0566913141999005) : 0.0115789118395218) : ((f11 > 0.0002500667) ? ((f26 > 0.1297914) ? -0.0820357413936745 : ((f19 > 0.2271409) ? 0.113267961472752 : -0.113996404262008)) : ((f31 > 18.5) ? 0.076650829795452 : ((f13 > 21.11911) ? -0.105663660602513 : ((f12 > 3.893767) ? ((f29 > 2.5) ? ((f23 > 1.631086) ? ((f0 > 2.36744) ? ((f31 > 8.5) ? -0.0792071137959855 : 0.391427749672784) : -0.151398830626765) : ((f4 > 0.04222184) ? ((f32 > 0.2002445) ? ((f31 > 6.5) ? 0.238890167467988 : ((f2 > 0.8832088) ? 0.650129176270518 : -0.246703778154982)) : ((f9 > 15.97174) ? -0.0648296373343517 : 0.031043977152899)) : ((f19 > 2.775509) ? -0.017334745047105 : ((f26 > 0.1486437) ? ((f0 > 1.170698) ? ((f32 > 0.05634726) ? 0.25814158216468 : 0.111380611585977) : 0.0531809638608805) : 0.372712688254656)))) : ((f30 > 0.01886052) ? ((f23 > 0.3438162) ? -0.05068449269221 : ((f17 > 2.942679) ? -0.164951845568243 : 0.0562333671787605)) : ((f24 > 0.008522198) ? ((f23 > 0.234156) ? ((f29 > 1.5) ? ((f4 > 0.2217676) ? 0.185019968313311 : 0.0658757865071639) : 0.000874548358784197) : 0.132612092652504) : -0.0182164936276547))) : ((f27 > 0.4001538) ? 0.133932533939036 : ((f13 > 4.140955) ? -0.121696774600438 : ((f7 > 0.2799163) ? ((f6 > -0.01632359) ? -0.270465146268626 : ((f7 > 0.999862) ? ((f25 > 0.7106913) ? 0.817996176214243 : 0.31453552271335) : -0.115621784768838)) : ((f12 > 0.02852431) ? 0.0178985468962183 : ((f2 > 0.8442891) ? 0.881427198110078 : 0.455719200570158)))))))))));
            double treeOutput58 = ((f16 > 80.5) ? ((f9 > 3.675052) ? ((f19 > 1.205926) ? -0.0386362775879078 : ((f32 > 0.312164) ? 1.72136677133273 : -0.028482861693125)) : ((f31 > 37.5) ? 0.263999893178156 : 0.0265626763116928)) : ((f18 > 2.222041) ? ((f9 > 17.96898) ? -0.0513866289569558 : ((f25 > 0.5419895) ? ((f0 > 1.343876) ? ((f23 > 1.631086) ? -0.058704616904665 : ((f12 > 2.914878) ? ((f27 > 0.04813278) ? 0.220305577820755 : 0.0670964971256401) : ((f26 > 0.5001454) ? 0.213015325992515 : -0.0603437556208985))) : ((f4 > 0.5207987) ? ((f23 > 0.6214191) ? -0.268015847305182 : 0.0770558755664025) : -0.00121643059763985)) : ((f28 > 0.1222164) ? 0.0455020721568986 : ((f31 > 4.5) ? ((f27 > 0.3065697) ? 0.0136693189472307 : 0.230598595332615) : 0.0821533383156536)))) : ((f21 > 1.936216) ? 0.469397063427936 : ((f30 > 0.02720838) ? ((f6 > 0.2786338) ? -0.0677601012538922 : -0.00854808466826464) : ((f25 > 0.4369328) ? ((f23 > 0.1630079) ? ((f29 > 1.5) ? ((f24 > 0.04716515) ? ((f15 > 0.499135) ? ((f12 > 20.66683) ? -0.0829679884458719 : 0.145772059174521) : 0.0333939557035856) : -0.272170884206624) : ((f32 > 0.1626157) ? ((f2 > 0.7498534) ? -0.191568471593281 : ((f27 > 0.2053158) ? ((f23 > 0.5991147) ? -0.387710183588155 : ((f31 > 6.5) ? 0.127260649760537 : 0.484895957713471)) : -0.23413517197216)) : ((f28 > 0.003329541) ? ((f23 > 0.5505459) ? -0.0126954235280578 : 0.20852747982436) : -0.00746109481141505))) : ((f3 > 0.3998359) ? 0.0239804703121363 : ((f28 > 0.0162893) ? -0.0348065190558916 : ((f0 > 0.5356073) ? 0.0286316567596456 : ((f9 > 8.686263) ? 0.331146872799693 : ((f15 > 0.703305) ? 0.625963760976948 : 0.426438303113801)))))) : 0.000783675091718525)))));
            double treeOutput59 = ((f15 > 0.8311566) ? ((f24 > 0.3402435) ? 0.0454245433881556 : ((f26 > 0.5369684) ? ((f32 > 0.063823) ? 0.0326425033955638 : 0.446441817945098) : ((f17 > 5.799238) ? -0.201112043596142 : 0.0201157008023991))) : ((f24 > 1.476991) ? ((f5 > 0.5502803) ? ((f25 > 1.043569) ? ((f27 > 0.7585866) ? -0.225096794124749 : -0.72067412331319) : 0.307355324248255) : 0.173099295646054) : ((f30 > 0.02482514) ? ((f6 > 0.2786338) ? ((f18 > 2.903062) ? ((f9 > 10.37847) ? -0.166901854243918 : ((f14 > 0.06957048) ? ((f10 > 1.347377) ? 0.138396654836707 : -0.285913244877446) : -0.0176373858555831)) : ((f32 > 0.1332605) ? 0.010381640289795 : ((f26 > 0.3591796) ? -0.134494483094977 : -0.066774467535906))) : ((f4 > 0.7054959) ? -0.189440123664356 : ((f9 > 18.41715) ? ((f25 > 1.258936) ? ((f3 > 0.002851777) ? 0.455791838894302 : -0.305774799884838) : -0.0605848010370417) : ((f32 > 0.1027937) ? ((f22 > 0.361447) ? -0.00223814945316855 : ((f14 > 0.06155303) ? 0.219454385533328 : ((f6 > -0.1174213) ? 0.00272189091282821 : 0.182712501101602))) : ((f22 > 1.823735) ? -0.630967243845378 : -0.0129057427347107))))) : ((f29 > 2.5) ? ((f16 > 177.5) ? -0.0361009121892956 : ((f18 > 1.302372) ? ((f11 > 3.093507) ? ((f6 > 0.0343403) ? ((f12 > 23.03892) ? 0.382276679949893 : 0.141723111267641) : 0.152918595717297) : ((f15 > 0.4878115) ? ((f16 > 105.5) ? 0.190551256421062 : 0.40289529471355) : ((f15 > 0.08099939) ? ((f29 > 5.5) ? 0.24917270879921 : ((f16 > 83.5) ? -0.0210915008386699 : ((f15 > 0.3741958) ? 0.397570764026642 : 0.105964405023596))) : -0.111451810193102))) : ((f26 > 0.1509575) ? -0.0286263873961798 : 0.136474209970973))) : -0.0159471099463421))));
            double treeOutput60 = ((f15 > 0.3225682) ? ((f7 > -0.2483636) ? ((f7 > -0.03892202) ? ((f12 > 5.85511) ? ((f12 > 7.238351) ? ((f13 > 0.05567709) ? 0.00147171862709235 : 0.355409193274831) : ((f12 > 6.932737) ? ((f12 > 7.032344) ? 0.0994813470818209 : ((f31 > 3.5) ? 0.26678805777178 : 0.25429352991452)) : 0.0429335465633235)) : -0.0636531423999662) : ((f12 > 0.02852431) ? ((f10 > 2.321455) ? ((f9 > 18.41715) ? ((f20 > 1.637396) ? ((f32 > 0.04256207) ? ((f19 > 2.409425) ? ((f25 > 0.684077) ? ((f28 > 0.1249407) ? 0.222948594787632 : ((f0 > 1.858214) ? 0.106632007375031 : ((f23 > 0.9603764) ? ((f1 > 0.9519409) ? -0.0193224892397987 : -0.25673999939507) : -0.0316643664684267))) : -0.18252081253382) : ((f6 > -0.07182601) ? 0.177419916504119 : -0.050251541809903)) : ((f31 > 3.5) ? 0.60651001300888 : 0.120905259390095)) : ((f6 > -0.05584227) ? ((f1 > 0.9190375) ? -0.277673552270213 : ((f25 > 0.4609665) ? ((f5 > 0.1810139) ? 0.0156917206238273 : 0.314269484272134) : 0.176007293437445)) : ((f25 > 0.6344197) ? ((f0 > 0.7074611) ? 0.0159133979034545 : ((f29 > 1.5) ? 0.329270870735751 : 0.0625861215952246)) : -0.0160452679107735))) : ((f14 > 0.3973684) ? ((f32 > 0.08649219) ? 0.222880532450664 : 0.104895867360827) : ((f13 > 0.9012095) ? ((f23 > 1.554531) ? -0.0797515025374109 : 0.0419394463930017) : ((f12 > 4.111117) ? 0.0210096493602882 : ((f24 > 0.02052259) ? -0.278833533486753 : -0.0940600017678675))))) : -0.0969607891652467) : 0.56874837863188)) : -0.0247163714453762) : ((f30 > 0.02337446) ? ((f1 > 1.275451) ? ((f4 > 0.6307976) ? -0.14893168524918 : 0.0853634487479818) : ((f6 > 0.3488724) ? -0.12917139809995 : -0.0578819956151526)) : -0.0316654528596379));
            double treeOutput61 = ((f16 > 54.5) ? ((f31 > 16.5) ? ((f2 > 0.02528848) ? ((f25 > 0.3739396) ? ((f20 > 1.305732) ? ((f29 > 10.5) ? ((f5 > 4.309524E-05) ? 0.0280725456476498 : ((f0 > 1.434483) ? 0.00178556802343232 : -0.345125177273389)) : ((f30 > 0.00746522) ? 0.0889686721419634 : 0.294973554369067)) : ((f19 > 2.47036) ? -0.172234765252239 : ((f18 > 1.838156) ? ((f28 > 0.1477007) ? ((f14 > 0.1746758) ? 0.0737566103447096 : -0.257190756020555) : 0.283037759251862) : -0.057272170225837))) : ((f18 > 0.5832889) ? ((f2 > 0.09407881) ? ((f30 > 0.03390822) ? 0.0945933651444831 : 0.316039455004436) : 0.00230778792316468) : -0.0641288964257971)) : -0.38265890456683) : ((f27 > 0.07208133) ? ((f13 > 8.127401) ? -0.04305369462731 : ((f12 > 8.142876) ? ((f14 > 0.04040816) ? 0.00872128483822077 : -0.145428223101284) : ((f14 > 0.2519841) ? -0.0424715099650481 : ((f25 > 0.6499294) ? 0.101794416285919 : 0.0185305695510198)))) : ((f29 > 4.5) ? 0.0482398086882658 : ((f3 > 0.4105584) ? ((f25 > 0.8325248) ? ((f23 > 0.6328585) ? -0.0332751089542384 : ((f25 > 1.158015) ? 0.512505997810466 : 0.133359112957436)) : ((f30 > 0.0571638) ? -0.263031357612954 : -0.0782858433828267)) : ((f1 > 0.01355108) ? ((f17 > 1.148987) ? ((f0 > 1.745745) ? -0.211731305857625 : ((f32 > 0.1281802) ? 0.196756831426086 : ((f3 > 0.1206793) ? ((f23 > 0.9979342) ? 0.13358190976589 : -0.0352216756063016) : 0.0885387075024349))) : ((f28 > 0.01958055) ? ((f25 > 0.03439728) ? ((f24 > 0.05856755) ? 0.0437980047735426 : -0.31216178173598) : 1.06732264718096) : ((f31 > 3.5) ? 0.205773461839647 : 0.153991763114533))) : ((f2 > 1.356579) ? 0.876521141302159 : -0.0838930154838363)))))) : 0.0325962416084151);
            double treeOutput62 = ((f24 > 1.733555) ? 0.167098701893675 : ((f30 > 0.153958) ? ((f1 > 0.6605569) ? -0.143003124560773 : 1.08946980988159) : ((f18 > 3.896952) ? ((f28 > 0.487019) ? -0.233898981085287 : ((f5 > 4.309524E-05) ? ((f32 > 0.0294245) ? ((f28 > 0.2049196) ? ((f13 > 0.9012095) ? 0.0609613621226399 : ((f32 > 0.1738583) ? -0.498227536998357 : 0.0389709690619955)) : ((f23 > 0.8933209) ? 0.13892515234684 : ((f4 > 0.3176587) ? 0.514106594702715 : 0.204129353210807))) : 0.595400010941045) : ((f27 > 0.09089838) ? -0.0814393576916875 : ((f26 > 0.244778) ? ((f32 > 0.179564) ? 0.395128743962061 : ((f26 > 1.080976) ? -0.249815349597264 : ((f23 > 1.631086) ? ((f19 > 2.286003) ? 0.0970297567945336 : -0.447692381717715) : ((f2 > 0.8188744) ? 0.0782322403600427 : ((f30 > 0.1120599) ? -0.114460487534692 : ((f3 > 0.4069175) ? 0.262316073312215 : ((f30 > 0.09474178) ? 0.355694906007194 : 1.12284403386264))))))) : -0.0791435073228143)))) : ((f30 > 0.05266714) ? ((f26 > 0.02499962) ? ((f32 > 0.179564) ? ((f27 > 0.3436125) ? 0.294016587867056 : 0.0185829138042953) : ((f4 > 0.3862948) ? -0.189488302751843 : -0.0418325233925054)) : ((f6 > -0.131271) ? 0.207534261854599 : 1.25717434611615)) : ((f28 > 0.2766749) ? 0.138350051624095 : ((f29 > 2.5) ? ((f23 > 0.1460426) ? ((f18 > 1.127722) ? ((f16 > 148.5) ? ((f31 > 46.5) ? 0.371460088787025 : -0.0219371008206991) : ((f15 > 0.5196805) ? ((f30 > 0.02880905) ? 0.0648386532231966 : 0.221392640025782) : ((f30 > 0.03390822) ? -0.0330860080492282 : 0.0691573355802291))) : ((f3 > 0.1319933) ? -0.0480710918068117 : 0.119421713566026)) : -0.186695189275502) : ((f30 > 0.01723713) ? -0.0356896081611074 : 0.00134558137169999)))))));
            double treeOutput63 = ((f16 > 96.5) ? ((f7 > 0.9878271) ? 0.114923798017928 : ((f6 > 0.7911065) ? 0.104510054864923 : ((f14 > 0.5125078) ? -0.162739808659321 : ((f30 > 0.1022561) ? -0.133324501970443 : ((f29 > 7.5) ? ((f1 > 1.071631) ? ((f4 > 0.01397852) ? 0.109010176829192 : 0.339127931105111) : ((f30 > 0.04394639) ? -0.0707910050801024 : ((f32 > 0.08326913) ? 0.175787357867376 : 0.0139497933624787))) : ((f30 > 0.02796866) ? ((f3 > 0.7544315) ? ((f25 > 0.9742545) ? -0.0349439387142787 : -0.241399609983133) : -0.0431801513712297) : -0.0243243831877299)))))) : ((f31 > 9.5) ? ((f32 > 0.179564) ? ((f29 > 2.5) ? ((f13 > 1.813096) ? ((f25 > 1.258936) ? -0.397393116712664 : 0.134480006389628) : ((f22 > 0.3997788) ? -0.2363657845082 : 0.389932957787363)) : ((f28 > 0.2549152) ? 0.0825793915462627 : -0.163763539061783)) : ((f24 > 0.07550176) ? ((f4 > 0.002237685) ? ((f27 > 0.3629766) ? -0.0485575858352558 : ((f18 > 2.726756) ? ((f26 > 0.7179694) ? ((f0 > 0.9248424) ? 0.112480637544971 : -0.432385198615468) : 0.168014611873919) : ((f5 > 0.05199241) ? ((f25 > 0.08438955) ? -0.010016134181888 : ((f3 > 0.1433548) ? 0.356470343595017 : -0.0719918467631657)) : ((f19 > 2.146651) ? ((f26 > 0.2762148) ? 0.072508202083258 : ((f29 > 5.5) ? 0.297199948990698 : -0.152686345318358)) : ((f17 > 2.821601) ? 0.236462306364097 : ((f4 > 0.230332) ? -0.112666214391088 : 0.1199178367678)))))) : ((f12 > 10.88849) ? 0.0827823508830846 : -0.215619836026716)) : ((f25 > 0.6047524) ? ((f0 > 0.6370065) ? 0.200376531057216 : 0.653796540108027) : ((f23 > 0.1335747) ? -0.0787880577526903 : 0.196395860562278)))) : ((f28 > -0.0002392505) ? 0.00434472527169112 : -0.15738830603272)));
            double treeOutput64 = ((f15 > 0.6063874) ? ((f31 > 8.5) ? ((f10 > 18.05156) ? ((f32 > 0.06087447) ? ((f13 > 0.8539063) ? ((f23 > 0.3602791) ? ((f5 > 0.1850025) ? 0.0557038024999858 : ((f6 > -0.1174213) ? -0.0557686648387832 : ((f9 > 17.44275) ? -0.0228464363717765 : 0.200597904361245))) : ((f21 > 0.5582675) ? ((f18 > 2.308783) ? 0.287817206832602 : -0.0563489028901427) : 0.129563071925929)) : -0.198226319214141) : ((f2 > 0.500604) ? 0.360281926875556 : 0.0975274417432894)) : ((f16 > 177.5) ? -0.00525068053410843 : ((f32 > 0.04760239) ? ((f10 > 1.522427) ? ((f16 > 80.5) ? 0.0269082163678826 : ((f24 > 0.112229) ? ((f32 > 0.1000589) ? 0.100613082412787 : 0.214626144908383) : 0.00671105019934405)) : -0.0220006469410446) : ((f1 > 0.1797143) ? ((f9 > 7.36611) ? 0.216136234267615 : ((f31 > 11.5) ? 0.607182741027865 : 0.367783968145227)) : 0.0627123437919849)))) : 0.00565157207733269) : ((f21 > 0.2475448) ? ((f10 > 3.349904) ? ((f32 > 0.03347) ? ((f31 > 5.5) ? ((f7 > 0.1055546) ? ((f10 > 12.9117) ? -0.256524643522359 : -0.0407522047990124) : ((f2 > 0.510906) ? 0.0287907901217892 : ((f23 > 1.175754) ? -0.182676119870245 : -0.0107089966629843))) : -0.108555729461383) : ((f18 > 1.902291) ? ((f9 > 11.31238) ? 0.0367808491599697 : ((f14 > 0.3009434) ? -0.00685085114502412 : 0.365895303433945)) : 0.0286666944527314)) : ((f7 > 0.1803339) ? ((f10 > 1.980432) ? 0.111765305031499 : -0.0485792204747076) : -0.10021722602011)) : ((f21 > 0.004402025) ? ((f26 > 0.1818348) ? ((f20 > 2.248165) ? 0.115384016619987 : -0.0256815230618551) : ((f29 > 3.5) ? ((f32 > 0.03929406) ? 0.246096539576809 : 0.127623452178071) : 0.0641924617863578)) : -0.0369024306540238)));
            double treeOutput65 = ((f6 > -0.2156612) ? ((f15 > 0.1091983) ? ((f16 > 361.5) ? -0.0873727386056542 : ((f31 > 24.5) ? ((f29 > 18.5) ? ((f28 > 0.04146852) ? 0.135453183555272 : -0.393562657457092) : 0.13189443729407) : ((f24 > 1.228056) ? 0.0933979824793148 : ((f30 > 0.08182147) ? -0.059128117638339 : ((f29 > 4.5) ? ((f28 > 0.07962343) ? ((f23 > 0.5505459) ? 0.0277785778714087 : -0.185736157386288) : ((f18 > 1.437583) ? ((f25 > 0.8051723) ? ((f0 > 1.65807) ? 0.228085131239791 : ((f23 > 0.5827338) ? ((f4 > 1.519449E-05) ? -0.0019112403025488 : -0.191465291768717) : 0.246625077589241)) : ((f27 > 0.23544) ? ((f17 > 3.403136) ? 0.146796825239964 : -0.116836673541549) : 0.14523704040795)) : -0.0142502568513857)) : ((f30 > 0.03487648) ? ((f23 > 0.5773079) ? ((f0 > 1.255128) ? ((f27 > 0.07295632) ? 0.0544468790428055 : ((f26 > 0.2850043) ? -0.111283500658642 : ((f23 > 0.9851624) ? 0.153774295907772 : -0.182451945335774))) : ((f27 > 0.2497463) ? -0.19231744314873 : -0.0641952558811567)) : 0.00688201599994443) : ((f29 > 1.5) ? ((f16 > 60.5) ? ((f30 > 0.01961497) ? -0.020744934221528 : ((f25 > 0.703735) ? ((f14 > 0.1297335) ? -0.0225306443395905 : 0.158006523797494) : 0.038843004643442)) : ((f18 > 2.089356) ? ((f15 > 0.7142262) ? ((f12 > 0.9221007) ? 0.402935398495005 : -0.111033644441297) : ((f25 > 0.4732654) ? 0.0615946624932773 : 0.27701623084845)) : ((f24 > 0.1178975) ? 0.0634389856149577 : -0.0522294553747023))) : ((f30 > 0.01265378) ? -0.0630658230696186 : ((f32 > 0.01084124) ? ((f23 > 0.3157126) ? -0.00225227076283489 : ((f30 > 0.006990564) ? 0.01867876776509 : 0.157556854900406)) : -0.0686304224212465))))))))) : -0.106055528377477) : -0.148384946487719);
            double treeOutput66 = ((f16 > 73.5) ? ((f9 > 4.45397) ? ((f10 > 18.29356) ? ((f7 > -0.1666668) ? ((f16 > 378.5) ? 0.473945523789939 : 0.0601783283556799) : ((f23 > 1.085162) ? -0.0771512605026698 : 0.00497173520616748)) : ((f7 > 0.9524139) ? 0.0687634795752241 : ((f19 > 1.168407) ? ((f10 > 2.591825) ? -0.0326066865065479 : -0.113163126622707) : ((f32 > 0.0754604) ? 0.154609697966872 : -0.0711413050554146)))) : ((f31 > 34.5) ? 0.175355285012369 : 0.0153639773501846)) : ((f1 > 0.8465301) ? ((f25 > 0.5565051) ? ((f0 > 1.477073) ? ((f26 > 0.46027) ? ((f26 > 0.795416) ? -0.00356656009640576 : ((f30 > 0.04445453) ? 0.159629441427467 : 0.322931339691246)) : ((f12 > 3.82842) ? ((f25 > 1.00577) ? -0.0699241048917751 : 0.177617683489615) : -0.110796691365962)) : ((f27 > 0.1090485) ? ((f31 > 9.5) ? 0.00961069613022005 : ((f19 > 4.119608) ? 0.141824773510841 : ((f14 > 0.01612903) ? -0.0899921172619584 : ((f23 > 0.3877518) ? ((f30 > 0.04001511) ? ((f5 > 0.1392462) ? -0.161978454276458 : -0.383647628534173) : -0.0925449830124625) : 0.100399360808184)))) : 0.0293535119883476)) : ((f17 > 5.223817) ? ((f4 > 0.3176587) ? 0.235330728509533 : -0.109881299537409) : ((f3 > 0.1461593) ? 0.174968648986429 : ((f28 > 0.2130173) ? -0.385154152053603 : 0.0338829618623036)))) : ((f27 > 0.5089349) ? 0.374456695017366 : ((f25 > 1.092159) ? 0.194631285976516 : ((f28 > 0.4203187) ? 0.742480373212085 : ((f4 > 0.3283691) ? ((f25 > 0.7509996) ? 0.222879223964215 : ((f2 > 0.298209) ? ((f30 > 0.01613588) ? -0.133387515785607 : -0.00571796464520139) : 0.324965129262773)) : ((f0 > 1.745745) ? ((f26 > 0.4852667) ? 0.310694531446262 : -0.170701467220987) : 0.0136870726546003)))))));
            double treeOutput67 = ((f15 > 0.3411807) ? ((f29 > 3.5) ? ((f30 > 0.059398) ? ((f6 > -0.1234417) ? ((f31 > 2.5) ? ((f18 > 3.05299) ? ((f30 > 0.153958) ? ((f26 > 0.2427617) ? ((f0 > 1.27053) ? 0.0112085237732196 : -0.300376885609221) : ((f1 > 2.029585) ? -0.565950718656143 : -0.0713005309061615)) : 0.0322293643907469) : -0.0593723127658124) : -0.253780559046049) : 0.0981679150384854) : ((f16 > 567.5) ? -0.109830351998552 : ((f1 > 0.6214421) ? ((f19 > 2.347493) ? ((f0 > 1.499442) ? 0.150196705059014 : -0.000528180693263968) : ((f27 > 0.01329084) ? ((f32 > 0.259126) ? -0.184192834816298 : ((f23 > 1.217493) ? 0.0208481404667317 : ((f27 > 0.1336371) ? ((f31 > 11.5) ? 0.237787117340543 : 0.0217474082563356) : 0.258192468794347))) : ((f32 > 0.09086977) ? 0.48439109110873 : 0.186681715526204))) : ((f30 > 0.03446827) ? ((f16 > 140.5) ? -0.166831836702379 : -0.00128485867710954) : ((f0 > 1.572526) ? 0.395884859436466 : ((f31 > 13.5) ? 0.113736853697359 : 0.0261382310398391)))))) : ((f30 > 0.01723713) ? ((f6 > 0.1748009) ? ((f9 > 11.91558) ? -0.13612714734588 : ((f18 > 0.9053625) ? ((f23 > 0.6046541) ? ((f17 > 3.932787) ? 0.0171122117046655 : -0.0783018803580439) : 0.0274955527141847) : -0.085250013543503)) : -0.00439597125203924) : ((f25 > 0.4609665) ? ((f6 > 0.1914617) ? ((f32 > 0.04039988) ? ((f4 > 0.4593562) ? 0.248451334840387 : ((f23 > 0.9036728) ? -0.101346749064406 : ((f5 > 0.1302615) ? 0.233060904265074 : 0.0221221882539057))) : ((f9 > 2.610359) ? ((f14 > 0.1680791) ? 0.0258992581112994 : 0.233175608283485) : ((f32 > 0.01315443) ? 0.291124250430849 : 1.29101585183341))) : 0.0284851738405519) : 0.00493829404213373))) : -0.0336968530629544);
            double treeOutput68 = ((f16 > 127.5) ? -0.0333198605505566 : ((f29 > 4.5) ? ((f30 > 0.07896698) ? -0.0195420031165103 : ((f24 > 0.3031055) ? ((f28 > 0.1412316) ? 0.0158419221752523 : ((f2 > 1.047977) ? ((f26 > 0.5743704) ? 0.153207119307598 : ((f26 > 0.1754448) ? ((f27 > 0.02059509) ? -0.0416609966322113 : -0.288410536905317) : 0.352877470342784)) : ((f31 > 19.5) ? 0.278961575186107 : ((f15 > 0.6481352) ? 0.17656257596841 : 0.0728661241124672)))) : -0.0366352954553953)) : ((f30 > 0.03772757) ? -0.0256915304971633 : ((f29 > 2.5) ? ((f15 > 0.4952324) ? ((f1 > 0.3121756) ? ((f16 > 74.5) ? ((f30 > 0.02577597) ? 0.0241417571355369 : 0.205390364925861) : 0.174734924953375) : -0.0130209594945055) : 0.0113749734012604) : ((f25 > 0.6779889) ? ((f28 > 0.1750538) ? ((f0 > 1.005143) ? 0.0189654899131356 : 0.340166890795273) : ((f23 > 0.4038999) ? ((f0 > 2.36744) ? ((f26 > 0.4395775) ? 0.472810797314567 : -0.109188675340754) : ((f32 > 0.05634726) ? -0.0380882489537135 : ((f31 > 2.5) ? 0.12256556590566 : -0.131632166148948))) : ((f25 > 0.9008789) ? ((f0 > 0.6798639) ? 0.649862253598222 : 0.148655836657951) : 0.0842250286820805))) : ((f3 > 0.5077426) ? ((f32 > 0.1110331) ? ((f19 > 2.532057) ? ((f3 > 0.5806987) ? 0.0389088984845522 : ((f22 > 0.2298346) ? ((f17 > 4.581218) ? -0.825487365695474 : 0.0611639108575648) : -0.288873816412427)) : ((f25 > 0.5613818) ? ((f1 > 0.4019715) ? -0.158974688750186 : 0.176444915842027) : -0.186480731403173)) : ((f4 > 0.1553751) ? ((f0 > 1.22584) ? 0.345084773210138 : 0.042982789371384) : -0.0894954952643454)) : ((f4 > 0.5207987) ? 0.628674894442978 : ((f17 > 5.507991) ? -0.195960698500726 : 0.00318926135650975))))))));
            double treeOutput69 = ((f15 > 0.7345737) ? ((f1 > 0.5284226) ? ((f25 > 0.3562232) ? 0.0211655993459058 : 0.122717795423166) : ((f25 > 0.4862599) ? ((f23 > 0.1277043) ? ((f32 > 0.03060906) ? ((f29 > 0.5) ? ((f14 > 0.9958677) ? ((f5 > 0.2610272) ? 1.13310939145707 : 0.245709556259537) : 0.0180681430229446) : ((f16 > 1013.5) ? 2.4637022538822 : -0.164922165522099)) : 0.241457324294411) : 0.166877126615648) : ((f3 > 0.3772435) ? -0.0994743773833003 : ((f0 > 0.5410861) ? -0.0417779797194092 : 0.0403743264223003)))) : ((f21 > 0.1720425) ? ((f24 > 0.1065523) ? ((f31 > 33.5) ? 0.155009118367385 : ((f30 > 0.01922392) ? ((f15 > 0.2806952) ? ((f0 > 1.120986) ? 0.0144773148275864 : ((f20 > 2.248165) ? ((f23 > 0.5611383) ? ((f0 > 0.8153569) ? -0.160387849535474 : -0.43877102349767) : -0.00385656292978519) : ((f1 > 0.9695498) ? ((f29 > 2.5) ? ((f22 > 1.197596) ? -0.194568444884691 : ((f23 > 0.9254424) ? -0.0323919579800461 : 0.231218291755906)) : -0.135380574280489) : ((f27 > 0.2329018) ? -0.108640563039128 : -0.0206242743721589)))) : -0.0807002765903971) : ((f10 > 2.203861) ? ((f28 > 0.002456723) ? ((f9 > 11.64751) ? 0.0148981524336027 : ((f16 > 258.5) ? -0.113852378436262 : ((f29 > 0.5) ? ((f11 > 2.098558) ? 0.196381132961419 : 0.0907307022581973) : 0.710783057659972))) : ((f2 > 1.240117) ? 0.406616130535122 : -0.0227230701545173)) : -0.0584383883247505))) : ((f30 > 0.009005889) ? ((f25 > 0.5904019) ? 0.104581443099799 : -0.149087082009182) : -0.0577933915602183)) : ((f21 > 0.004402025) ? ((f21 > 0.05178685) ? 0.0160838295399549 : ((f32 > 0.1000589) ? 0.262567878641546 : ((f28 > 0.01470524) ? 0.560908393206986 : 0.0978671891974177))) : -0.0243024289025106)));
            double treeOutput70 = ((f16 > 61.5) ? ((f9 > 6.808784) ? ((f7 > 1) ? 0.0904489002631943 : ((f14 > 0.6702899) ? -0.316980167012155 : -0.025575758350727)) : ((f18 > 4.381141) ? 0.0918996885199963 : ((f30 > 0.04764608) ? ((f26 > 0.1732638) ? ((f2 > 0.5162106) ? ((f32 > 0.1184021) ? 0.0411372976619868 : -0.0895866385914833) : ((f25 > 0.2318654) ? -0.19836164880509 : 0.179430065432779)) : ((f25 > 0.3056822) ? 0.414830494309854 : -0.0376199806872813)) : ((f19 > 3.415355) ? ((f31 > 4.5) ? ((f23 > 0.6328585) ? 0.0888226817616227 : 0.27125989637548) : -0.0354395963805657) : ((f20 > 2.691186) ? ((f0 > 1.005143) ? -0.0201082013627283 : -0.34812140852782) : ((f13 > 0.05567709) ? ((f16 > 287.5) ? -0.0787977540463856 : ((f31 > 15.5) ? ((f10 > 1.686399) ? ((f2 > 0.0618449) ? ((f25 > 0.4450171) ? ((f26 > 0.2941461) ? 0.105303867005301 : ((f29 > 4.5) ? 0.0466090582192984 : ((f21 > 0.2209852) ? -0.278182823243243 : 0.154616560856934))) : ((f1 > 0.1546523) ? ((f14 > 0.05064103) ? 0.296162910423777 : 0.0627621699719326) : ((f14 > 0.1708207) ? -0.179014312899076 : 0.145465856518081))) : -0.15455919898378) : -0.0807233283126184) : 0.00821218939281845)) : 0.2331531079112)))))) : ((f18 > 2.089356) ? ((f13 > 0.9012095) ? ((f31 > 9.5) ? ((f21 > 1.729506) ? -0.0954122324530277 : 0.125487132524428) : ((f30 > 0.06577675) ? -0.0520421080783984 : ((f0 > 1.415576) ? 0.142263348416205 : ((f25 > 0.4609665) ? -0.00166659015495179 : ((f26 > 0.1345578) ? 0.183748190081905 : -0.0438881042913321))))) : ((f10 > 17.77325) ? -0.235943287945569 : -0.0335876550234258)) : ((f26 > 0.6433989) ? ((f32 > 0.06152839) ? 0.0597194852027811 : 0.69351082227504) : 0.0110388975377474)));
            double treeOutput71 = ((f7 > -0.3814833) ? ((f17 > 0.3568271) ? ((f3 > 0.06579271) ? ((f11 > 6.045397) ? ((f10 > 9.904465) ? 0.0044317803737306 : 0.094317550602309) : ((f10 > 2.132226) ? ((f13 > 0.02541088) ? ((f9 > 18.42068) ? -0.0367615721943185 : ((f29 > 2.5) ? ((f6 > 0.2687221) ? ((f9 > 11.31238) ? -0.131366849627225 : ((f30 > 0.0217307) ? ((f14 > 0.2970843) ? ((f7 > 0.4632018) ? -0.160683351217673 : 0.114080238974696) : -0.0281076312073304) : 0.0783164804342638)) : 0.0402569785229379) : -0.00580296358666171)) : 0.256643437948654) : ((f7 > 0.1465143) ? -0.0167227731300094 : -0.111503048252913))) : ((f9 > 2.610359) ? ((f19 > 0.2536947) ? ((f28 > 0.05359487) ? 1.01701131223749 : ((f31 > 1.5) ? ((f1 > 1.691513) ? -0.212052636066573 : ((f30 > 0.04001511) ? ((f6 > -0.1503709) ? 0.250814671834219 : 0.724190498134399) : 0.2188145365989)) : 0.0086856788569618)) : ((f10 > 17.99229) ? ((f7 > -0.1799985) ? ((f6 > -0.09741524) ? ((f0 > 0.8565323) ? -0.0500143538379139 : ((f23 > 0.1812081) ? ((f1 > 0.1256852) ? -0.0365767502968554 : 1.87613597646517) : 0.648900063357315)) : 0.133914919040704) : 0.0918516945223472) : 0.0514333263955166)) : 0.415862903720719)) : ((f28 > 0.03484204) ? ((f25 > 0.04537791) ? -0.213081372232739 : 1.46324093092658) : ((f4 > 0.001325623) ? ((f2 > 0.768613) ? 0.188669999003484 : ((f26 > 0.3465821) ? -0.164608089594819 : 0.09588265718932)) : ((f3 > 0.6934437) ? -0.301690116314618 : -0.13023119018293)))) : ((f9 > 12.8538) ? -0.199913750549987 : ((f10 > 6.004582) ? ((f18 > 1.437583) ? ((f14 > 0.01612903) ? ((f14 > 0.3421603) ? 0.0137048278996172 : 0.275408943886297) : 0.027618257571139) : -0.0403240421830273) : -0.080178895176236)));
            double treeOutput72 = ((f32 > 0.3643579) ? ((f31 > 2.5) ? 0.0929527145552278 : 0.740630218881233) : ((f16 > 54.5) ? ((f31 > 15.5) ? ((f13 > 10.21148) ? -0.0162051673002444 : 0.0489740508782742) : ((f23 > 0.6046541) ? ((f0 > 0.959049) ? ((f23 > 0.8829889) ? ((f17 > 3.978246) ? ((f23 > 1.119045) ? ((f0 > 1.328517) ? 6.90417233218947E-05 : ((f32 > 0.04132648) ? -0.158433953764547 : 0.0643734486059405)) : 0.0776352854168913) : ((f19 > 2.224677) ? ((f32 > 0.04464701) ? -0.160710581763734 : -0.0602023438122172) : -0.00137695055336179)) : ((f26 > 0.43485) ? ((f0 > 1.075927) ? ((f2 > 1.009111) ? 0.0904452143165735 : ((f24 > 0.1235496) ? 0.50096652839987 : -0.0518903857807144)) : ((f23 > 0.7930689) ? -0.00422410002094022 : 0.238847897119705)) : ((f27 > 0.04312705) ? 0.0955086044287329 : -0.0259923069171948))) : ((f25 > 0.2760238) ? ((f21 > 1.224924) ? -0.201305889156936 : -0.0807722993314486) : ((f2 > 0.09005587) ? 0.0958624316351477 : -0.096905201138698))) : ((f25 > 0.6662829) ? ((f24 > 0.5355691) ? ((f17 > 2.295105) ? 0.135108885712229 : ((f3 > 0.8832791) ? -0.273564479823517 : -0.031564635294044)) : ((f26 > 0.6653943) ? ((f0 > 0.7303274) ? 0.575659312174596 : 0.156672731749388) : ((f28 > 0.1291643) ? 0.33664859759464 : ((f0 > 1.477073) ? ((f26 > 0.5369684) ? ((f24 > 0.0005822969) ? 0.372339186457908 : 6.49785115458306) : -0.168049809384975) : ((f9 > 10.04372) ? 0.0213156482089076 : ((f31 > 4.5) ? 0.166899343360558 : ((f32 > 0.05265083) ? -0.183755674105551 : 0.169007610546871))))))) : ((f3 > 0.676752) ? ((f0 > 0.7130982) ? 0.135598622172611 : -0.329647729322707) : ((f17 > 4.025797) ? -0.194918776619402 : -0.00452156282828016))))) : 0.0271237859041246));
            double treeOutput73 = ((f15 > 0.2452766) ? ((f31 > 8.5) ? ((f5 > -1.303895E-06) ? ((f19 > 6.043689) ? -0.241456895386571 : ((f12 > 28.72018) ? ((f11 > 9.21034) ? 0.09729574471623 : -0.107805553515055) : ((f32 > 0.05333724) ? ((f14 > 0.01612903) ? ((f22 > 0.05487286) ? ((f23 > 0.2924106) ? ((f1 > 0.2300517) ? ((f19 > 1.80645) ? ((f31 > 34.5) ? 0.32487175549332 : 0.00601787962160637) : 0.197727597207714) : ((f5 > 0.3210939) ? 0.782565400902535 : -0.140666760864484)) : ((f30 > 0.01612253) ? -0.221948326864878 : 0.177770632744138)) : ((f32 > 0.1110331) ? ((f30 > 0.09095263) ? 0.0358121956560788 : 0.384283882608202) : ((f4 > 0.001325623) ? 0.122502580858298 : -0.163684063828669))) : ((f6 > -0.08907885) ? ((f27 > 0.04312705) ? ((f13 > 0.8539063) ? ((f12 > 9.054589) ? ((f13 > 8.844531) ? ((f5 > 4.309524E-05) ? ((f23 > 0.3710835) ? -0.0116537027549487 : 0.129521442929684) : -0.0633778847505955) : -0.185787141424554) : ((f13 > 9.125596) ? -0.101466815984038 : 0.0396962259993772)) : -0.27552750554575) : ((f1 > 0.9351752) ? 0.0884710051140912 : ((f30 > 0.02337446) ? ((f2 > 0.6696985) ? ((f12 > 6.617101) ? -0.158092298211224 : -0.389105540179808) : ((f1 > 0.572624) ? 0.26321440823603 : -0.101245700552323)) : 0.0121619022750238))) : ((f9 > 17.44275) ? -0.00768348295589854 : 0.151385375488963))) : ((f14 > 0.3009434) ? -0.0606795102864385 : ((f2 > 0.09407881) ? 0.160975315938891 : 0.0150822093641118))))) : ((f2 > 0.4533024) ? 0.429702122733139 : ((f27 > 0.1090485) ? -0.17611916279597 : 0.28518694609848))) : ((f28 > -0.0002392505) ? -0.000189757888336018 : -0.175270366284646)) : ((f12 > 40.48455) ? ((f18 > 1.92418) ? 0.209380229792858 : 0.0593968999761407) : -0.0600221465058711));
            double treeOutput74 = ((f18 > 6.742136) ? 0.103037384771389 : ((f4 > 0.872744) ? ((f2 > 1.443784) ? -0.677100309382771 : -0.0998903075596859) : ((f16 > 103.5) ? -0.0256701543893035 : ((f31 > 12.5) ? ((f32 > 0.2305679) ? -0.112300051396095 : ((f24 > 0.05290641) ? ((f10 > 1.733314) ? ((f10 > 10.98777) ? ((f7 > 0.2799163) ? -0.263102627983036 : ((f23 > 0.4147288) ? 0.020106093451134 : 0.138309807646986)) : 0.0950455497914902) : -0.0242609560173696) : ((f23 > 0.03675002) ? ((f32 > 0.1017082) ? ((f28 > 0.07421038) ? 0.103477905761655 : -0.213685723527528) : -0.00926960376672139) : 0.796062304911176))) : ((f15 > 0.696207) ? ((f10 > 18.05156) ? -0.00870794775839864 : ((f31 > 4.5) ? ((f7 > 0.3085158) ? 0.00960842688301224 : ((f9 > 18.23667) ? -0.0985845817584129 : ((f18 > 0.9777497) ? ((f14 > 0.06504495) ? ((f9 > 9.368672) ? 0.196082894100809 : ((f30 > 0.007243992) ? 0.269858191447214 : 1.80303821665625)) : ((f30 > 0.03296186) ? ((f27 > 0.1996688) ? -0.114832204232921 : 0.0701099500086024) : ((f16 > 67.5) ? 0.0886128934745656 : 0.278989456501884))) : 0.0579945747032983))) : 0.0274162691331527)) : ((f22 > 0.1135459) ? ((f25 > 0.06802459) ? ((f3 > 0.0424804) ? -0.0583301825330417 : 1.31757176458673) : ((f3 > 0.1095511) ? 0.150811506628453 : -0.0507625443251189)) : ((f5 > 0.003619287) ? ((f24 > 0.02960413) ? ((f25 > 0.2433198) ? 0.0452228373727389 : ((f2 > 0.03518806) ? ((f1 > 0.204794) ? ((f2 > 0.08630696) ? ((f2 > 0.2027341) ? 0.392186110718862 : 0.774417040971659) : 0.243861813780621) : ((f29 > 0.5) ? 0.0856401391857426 : 0.541993435213242)) : ((f23 > 0.2103111) ? 0.024044788495283 : 1.50234635127988))) : -0.0965486312964854) : -0.0105139104240254)))))));
            double treeOutput75 = ((f7 > -0.2556373) ? ((f7 > -0.1129836) ? ((f10 > 8.240831) ? -0.0565534486566784 : ((f10 > 2.092505) ? ((f2 > 0.05996452) ? ((f14 > 0.3521463) ? ((f9 > 3.772537) ? -0.0710724560929108 : 0.175706594863202) : ((f31 > 4.5) ? ((f16 > 75.5) ? ((f30 > 0.03390822) ? -0.0287938996833828 : ((f9 > 8.064808) ? 0.037788292105968 : 0.125859501383996)) : ((f14 > 0.0412415) ? ((f9 > 13.93937) ? 0.033391248236716 : 0.301378547750666) : 0.105526398316008)) : 0.0287826599699527)) : -0.0627915064069361) : ((f7 > 0.6524868) ? ((f12 > 0.09397569) ? 0.0354777156289371 : -0.284763937398005) : -0.0515345145174469))) : ((f10 > 3.238678) ? ((f10 > 6.273683) ? ((f32 > 0.03558557) ? ((f6 > -0.1174213) ? ((f6 > 0.6659548) ? ((f29 > 1.5) ? -0.0425350061543216 : -0.131372166974487) : ((f13 > 0.8539063) ? ((f23 > 1.554531) ? -0.120739428203784 : -0.00242475636506012) : -0.125803724185974)) : ((f9 > 18.29306) ? 0.0174218789156654 : 0.143197621887236)) : ((f31 > 2.5) ? ((f10 > 17.42497) ? ((f32 > 0.02353454) ? ((f31 > 5.5) ? 0.291318313504654 : 0.0875256541847922) : ((f25 > 0.6294107) ? ((f24 > 0.001430059) ? 0.795877281428655 : 0.181853914530764) : ((f9 > 4.846413) ? 0.353783015300661 : 0.610830913274905))) : 0.0726956723821917) : 0.0867224337150008)) : ((f7 > -0.1814419) ? ((f25 > 0.4862599) ? 0.222624710521727 : 0.185638000004641) : 0.0365257703942895)) : -0.0687995865350058)) : ((f10 > 5.02382) ? ((f10 > 16.75527) ? -0.043450012362647 : ((f2 > 0.07945781) ? ((f9 > 10.37847) ? 0.0126774780634333 : ((f31 > 4.5) ? ((f16 > 83.5) ? 0.101769635499872 : 0.192422729250695) : 0.0645121714114075)) : -0.102606999269579)) : -0.0830591861392882));
            double treeOutput76 = ((f1 > 1.49885) ? ((f4 > 0.7054959) ? ((f20 > 2.430951) ? ((f1 > 2.366133) ? ((f23 > 1.952857) ? ((f2 > 1.561375) ? 0.405511960207962 : -0.256428801484009) : 0.246202636801663) : -0.230720619858101) : 0.326705118498591) : 0.0829740037060156) : ((f30 > 0.04542997) ? ((f26 > 0.108036) ? ((f32 > 0.1332605) ? ((f2 > 0.5270088) ? 0.0386078253297567 : ((f3 > 0.4464557) ? 0.204447032315389 : -0.130641524840078)) : ((f3 > 0.5565874) ? ((f14 > 0.1043557) ? -0.0176486837721127 : -0.12206660262107) : ((f1 > 0.9519409) ? ((f23 > 0.4617791) ? ((f30 > 0.08821622) ? -0.106152089350532 : ((f25 > 0.8183883) ? -0.115742549225625 : ((f27 > 0.1751301) ? 0.0455738187223677 : ((f1 > 1.144978) ? ((f27 > 0.007282402) ? 0.337282752770728 : 0.739495563051762) : 0.175401477529307)))) : ((f21 > 1.072368) ? ((f26 > 0.3238384) ? 0.290082436772398 : 1.25838394010595) : 0.124758969908965)) : -0.0549180650017898))) : ((f2 > 0.133058) ? ((f6 > -0.190401) ? 0.226664515387921 : 0.671711137497382) : -0.110240547432478)) : ((f29 > 3.5) ? ((f1 > 0.358116) ? ((f16 > 149.5) ? ((f17 > 7.335512) ? 0.492199729640548 : ((f31 > 42.5) ? 0.29036181362783 : -0.00414784124192743)) : ((f15 > 0.1193988) ? ((f30 > 0.02720838) ? 0.0591834500679103 : 0.171822922859352) : ((f12 > 26.00829) ? 0.118191360257994 : -0.111932264686695))) : ((f25 > 0.7255468) ? 0.271868003576771 : -0.0401348843189679)) : ((f23 > 0.5664996) ? ((f17 > 2.993553) ? ((f23 > 0.8094885) ? -0.0148599259951486 : ((f27 > 0.1246383) ? 0.214561368428921 : 0.0268854760792185)) : -0.0676569880084445) : ((f28 > 0.1249407) ? ((f0 > 0.308156) ? 0.1516974344009 : -0.137199177283314) : 0.000840216876396917)))));
            double treeOutput77 = ((f15 > 0.8868023) ? 0.0251080647938111 : ((f18 > 5.619127) ? ((f30 > 0.1850781) ? -0.114900682144215 : ((f5 > 0.5502803) ? ((f2 > 0.775267) ? -0.242623656339996 : 0.227541557752264) : 0.116204245163581)) : ((f30 > 0.02423923) ? ((f32 > 0.1027937) ? ((f4 > 0.01559313) ? -0.00271235100059868 : ((f2 > 0.6696985) ? ((f1 > 0.8599844) ? ((f32 > 0.1626157) ? ((f31 > 2.5) ? 0.186312547664731 : ((f29 > 3.5) ? 0.0209996731984632 : 1.12613138999446)) : 0.0587080823764371) : ((f20 > 2.934237) ? -0.27003643755075 : 0.0146212419264345)) : ((f18 > 1.902291) ? ((f23 > 1.37829) ? ((f29 > 5.5) ? 0.556645988697479 : -0.451636963809004) : 0.485515371624279) : 0.0707778276954843))) : ((f6 > 0.1748009) ? ((f18 > 2.686128) ? ((f20 > 2.025493) ? -0.0989307288414259 : ((f31 > 5.5) ? 0.0777071143604657 : -0.0706353273026502)) : -0.0663641112123038) : -0.0146572582535902)) : ((f29 > 1.5) ? ((f14 > 0.1593651) ? ((f13 > 2.451372) ? -0.0690729501065637 : 0.057944141754288) : ((f19 > 1.882107) ? ((f29 > 3.5) ? ((f15 > 0.2005514) ? 0.213056952607551 : 0.0743998171654894) : ((f16 > 56.5) ? ((f30 > 0.01563009) ? ((f16 > 102.5) ? -0.103367850038127 : ((f15 > 0.5159602) ? ((f32 > 0.1864846) ? -0.0881257197386734 : 0.127905216281673) : -0.0252041859402499)) : ((f15 > 0.2883883) ? 0.155628507187053 : 0.0298460796979239)) : 0.147639701999203)) : ((f19 > 1.205926) ? -0.0313585687194664 : ((f32 > 0.07447337) ? 0.208856500633183 : ((f17 > 3.215649) ? ((f26 > 0.1881885) ? 0.322400426716445 : 0.0653844510414315) : 0.0380837530403196))))) : ((f30 > 0.01030548) ? -0.0643501270550057 : ((f7 > 0.9878271) ? 0.195837025630429 : -0.00487299093535441))))));
            double treeOutput78 = ((f16 > 69.5) ? ((f29 > 5.5) ? ((f30 > 0.08821622) ? -0.065842594632502 : ((f16 > 426.5) ? -0.066066949426108 : ((f28 > 0.106931) ? ((f1 > 0.6949899) ? ((f26 > 0.244778) ? 0.0155359999533517 : ((f2 > 0.3742337) ? 0.598299056163271 : 0.215031159671294)) : -0.103466865080461) : ((f5 > 4.309524E-05) ? ((f23 > 0.3211957) ? ((f1 > 0.4069912) ? ((f27 > 0.209097) ? 0.103261979066343 : 0.242008304770616) : 0.0476611674568293) : -0.140577886718583) : ((f32 > 0.087467) ? 0.0821915983829214 : -0.0217327021618464))))) : ((f30 > 0.02082193) ? ((f32 > 0.312164) ? ((f2 > 0.3962604) ? ((f3 > 0.2748175) ? ((f2 > 0.6943492) ? 0.124195863162829 : 0.591054441808759) : -0.384522280870898) : 1.34630143039787) : ((f20 > 2.430951) ? -0.0926784978418691 : ((f32 > 0.1626157) ? ((f25 > 0.7106913) ? ((f16 > 237.5) ? 0.780257200255745 : 0.138980005438409) : ((f31 > 3.5) ? ((f21 > 0.3394775) ? -0.0216216398619429 : ((f1 > 0.2768312) ? 0.33914394995034 : -0.167411476438927)) : -0.349841179231859)) : -0.0381694491414737))) : ((f14 > 0.1441558) ? -0.0724655103835414 : ((f25 > 0.9464693) ? ((f4 > 0.002237685) ? 0.198411709866819 : 0.045204837302073) : ((f29 > 1.5) ? 0.0370819399109513 : ((f30 > 0.009897223) ? -0.0896873780502649 : 0.00598518200761717)))))) : ((f31 > 4.5) ? ((f32 > 0.05562594) ? ((f9 > 1.875458) ? 0.00703161474533313 : -0.0948608858976999) : ((f15 > 0.6583485) ? ((f25 > 0.5807843) ? 0.434297530699577 : ((f18 > 0.7923127) ? ((f30 > 0.009170808) ? 0.175020611777217 : 1.04887716878428) : 0.0457593836583994)) : ((f6 > -0.05401162) ? ((f18 > 1.385967) ? 0.107278065532031 : 0.0341166601880067) : -0.0373244253369228))) : -0.00254936089484078));
            double treeOutput79 = ((f15 > 0.4137791) ? ((f29 > 3.5) ? ((f30 > 0.06155282) ? -0.00706769466712622 : ((f16 > 600.5) ? -0.0918887977799284 : ((f29 > 8.5) ? 0.158236517693387 : ((f16 > 177.5) ? ((f30 > 0.02648656) ? -0.11770573458357 : 0.0524048605783249) : ((f24 > 0.2311896) ? ((f30 > 0.02221101) ? 0.056340279752379 : 0.292678304605497) : -0.0398142578839729))))) : ((f30 > 0.03296186) ? -0.0249782613964631 : ((f29 > 2.5) ? ((f16 > 102.5) ? 0.00859248815538809 : ((f26 > 0.3865668) ? 0.228758493424619 : ((f1 > 0.4761062) ? 0.154168405269733 : 0.00852180832346849))) : ((f30 > 0.0136383) ? ((f29 > 1.5) ? ((f16 > 84.5) ? -0.0618572608284369 : ((f1 > 0.5470313) ? ((f15 > 0.5891528) ? ((f30 > 0.02512359) ? ((f16 > 65.5) ? -0.0064462159937845 : 0.157664745910123) : ((f12 > 0.8754225) ? 0.314860460245633 : -0.0848924005012417)) : 0.0152854251983959) : ((f26 > 0.5001454) ? 0.217023755508138 : ((f24 > 0.04141095) ? ((f30 > 0.02683053) ? -0.0532780255290599 : 0.0328115432432432) : -0.173688134175005)))) : -0.0559744998556123) : ((f24 > 0.0670104) ? ((f5 > 0.00509081) ? ((f28 > 0.05006097) ? 0.0526912039388927 : ((f29 > 0.5) ? 0.139615331955605 : ((f24 > 0.1982281) ? 1.59755935779368 : 0.375154429584569))) : ((f16 > 91.5) ? -0.0578196322519433 : ((f31 > 6.5) ? ((f15 > 0.751943) ? ((f0 > 0.7418712) ? 0.569992653742075 : 0.171628466530172) : 0.101230743664236) : ((f25 > 1.00577) ? 0.354726564335348 : 0.0270794375211341)))) : ((f32 > 0.0989999) ? -0.118488041213759 : ((f2 > 0.438556) ? ((f32 > 0.02458691) ? 0.0124204892646185 : ((f10 > 0.01398623) ? 0.322831065802416 : 1.08774309013682)) : 0.000941709484996721))))))) : -0.0227052444556671);
            double treeOutput80 = ((f16 > 158.5) ? ((f32 > 0.148212) ? ((f1 > 0.8882377) ? ((f3 > 0.5182701) ? 0.183537097885698 : -0.106609479940181) : ((f4 > 1.519449E-05) ? 0.289019116348609 : -0.00492765464208441)) : -0.0504618229357964) : ((f9 > 11.31238) ? -0.0123151455259003 : ((f10 > 1.796722) ? ((f31 > 7.5) ? ((f14 > 0.035401) ? ((f1 > 0.1755778) ? ((f25 > 0.4775596) ? ((f1 > 0.7968202) ? ((f25 > 0.6243948) ? 0.0692744809200335 : 0.285053597869184) : ((f5 > 0.006319606) ? ((f3 > 0.3930644) ? ((f30 > 0.01492007) ? -0.0237558018096418 : 0.302624351871747) : -0.143020433270558) : ((f29 > 3.5) ? 0.0132142901783716 : 0.284043897326655))) : ((f11 > 2.021782) ? 0.338036771896752 : ((f15 > 0.2572331) ? ((f23 > 0.1221177) ? 0.209563313939288 : -0.127752435575789) : 0.0638635451005776))) : ((f30 > 0.009254977) ? -0.134067960502743 : 0.061796152249955)) : ((f23 > 0.5935957) ? ((f32 > 0.04760239) ? ((f27 > 0.04813278) ? ((f26 > 0.1509575) ? ((f0 > 0.8493881) ? ((f23 > 0.7548721) ? -0.00610475363345115 : 0.121475007413683) : ((f26 > 0.5894637) ? -0.417954755909566 : ((f25 > 0.6662829) ? 0.185404952611252 : -0.109018815177225))) : -0.189155951152569) : ((f2 > 0.6696985) ? ((f1 > 0.7634702) ? -0.0618693928209763 : -0.237342649334841) : ((f32 > 0.1052382) ? 0.337505103064012 : -0.0203076906307596))) : 0.0882059746611478) : ((f25 > 0.3911977) ? ((f28 > 0.07508917) ? ((f3 > 0.4105584) ? 0.0827308765303871 : 0.311025573307427) : ((f32 > 0.1017082) ? -0.0321532002774526 : ((f17 > 4.765248) ? -0.167693797935395 : ((f26 > 0.2549882) ? 0.139580731147853 : 0.00702915913955993)))) : -0.0168666249881778))) : 0.0140436269409584) : ((f2 > 0.2165873) ? -0.0595613740425503 : 0.0489646425541525))));
            double treeOutput81 = ((f15 > 0.08699) ? ((f13 > 24.19214) ? ((f32 > 0.03323481) ? ((f12 > 4.344763) ? ((f12 > 23.77955) ? -0.00785707479493868 : -0.1762047112018) : 0.131114524890893) : -0.0507218618693752) : ((f11 > 10.71776) ? ((f1 > 0.02280172) ? ((f14 > 0.04040816) ? ((f16 > 137.5) ? 0.0177829353511646 : ((f9 > 9.028026) ? 0.15167140479498 : 0.305287406496538)) : 0.031506067573846) : 0.00877259798810347) : ((f24 > 1.733555) ? 0.160525349778588 : ((f12 > 3.82842) ? ((f13 > 0.05567709) ? ((f13 > 3.92349) ? ((f13 > 8.201846) ? ((f30 > 0.1270167) ? -0.133739476058567 : ((f12 > 7.894728) ? ((f14 > 0.1550481) ? ((f16 > 107.5) ? -0.243185749422149 : 0.0295053511628044) : ((f29 > 2.5) ? ((f4 > 0.01641226) ? ((f23 > 0.435869) ? 0.00650164929508741 : 0.0855177627194553) : ((f3 > 0.6330515) ? ((f23 > 0.2685121) ? -0.00178471702480563 : -0.36315713488864) : ((f32 > 0.07828732) ? 0.179578808251894 : 0.0836246523516722))) : 0.00623365813133477)) : -0.0625725390967698)) : ((f12 > 8.229173) ? -0.0268984292985036 : ((f25 > 0.4530378) ? ((f14 > 0.2313266) ? -0.0983229906011218 : ((f13 > 5.87136) ? ((f12 > 5.920735) ? ((f7 > 0.9524139) ? 0.0455001213748778 : ((f10 > 17.49965) ? 0.0891586439757974 : 0.282285349345474)) : 0.0477816560890877) : ((f13 > 4.081036) ? ((f7 > -0.2180184) ? 0.00540734188163872 : 0.142881395422957) : 0.218766727575006))) : 0.0633263259121912))) : ((f14 > 0.01612903) ? -0.00609615968732667 : ((f32 > 0.0468849) ? -0.132768910045407 : -0.0794081745728262))) : 0.223814547955638) : ((f26 > 0.5607868) ? ((f10 > 0.05218575) ? 0.0493680866956298 : ((f6 > -0.03778917) ? -0.129724172245431 : 0.604108213810803)) : -0.030090234687617))))) : -0.123640844011762);
            double treeOutput82 = ((f16 > 54.5) ? ((f32 > 0.3643579) ? ((f0 > 1.075927) ? ((f25 > 0.9223884) ? 0.369445432062726 : -0.0290970322119334) : 0.241043390962467) : ((f31 > 20.5) ? ((f6 > 0.6823342) ? 0.208823114705758 : 0.0206712735886693) : ((f2 > 0.2446529) ? ((f2 > 0.510906) ? ((f30 > 0.008925916) ? -0.00814771957709783 : ((f6 > 0.2264785) ? 0.111852873190464 : 0.0111591994051917)) : ((f6 > 0.6505057) ? ((f32 > 0.03622883) ? -0.146179737389849 : 0.0298971149629719) : ((f18 > 4.381141) ? 0.259506980653128 : -0.0397354695327363))) : ((f24 > 0.007212347) ? ((f25 > 0.04537791) ? ((f28 > 0.02219445) ? ((f24 > 0.08673937) ? ((f29 > 1.5) ? ((f26 > 0.1600224) ? ((f1 > 0.2768312) ? 0.178230539772676 : -0.130203841006298) : ((f26 > 0.04043211) ? ((f30 > 0.01852632) ? ((f14 > 0.03738644) ? -0.0716512267658782 : -0.245847974553485) : 0.0418799680818659) : 0.916018864438819)) : ((f31 > 7.5) ? 0.545059591375436 : 0.272716512439629)) : ((f30 > 0.006251954) ? -0.326829507858179 : -0.0230965187474675)) : ((f5 > 0.00509081) ? ((f24 > 0.06140137) ? ((f16 > 194.5) ? 0.135116266482408 : ((f27 > 0.02782674) ? 0.52621577277704 : 0.187537527012041)) : -0.0197503824726943) : ((f32 > 0.05562594) ? ((f2 > 0.09828465) ? 0.309814643376368 : -0.00894573842662312) : ((f3 > 0.07084431) ? 0.0560549532493896 : 0.215325998781195)))) : ((f23 > 0.2536597) ? ((f28 > 0.007058793) ? 0.0509991423039392 : -0.152011155727781) : ((f32 > 0.01954288) ? ((f5 > 0.006319606) ? ((f1 > 0.03579544) ? 0.918688070264801 : 0.0793376418523183) : 0.479662322802926) : ((f10 > 18.11803) ? ((f7 > -0.1876039) ? 1.24373603281883 : 0.487340328479476) : 0.280761877501136)))) : -0.0825128301060805)))) : 0.0235626840051748);
            double treeOutput83 = ((f15 > 0.53378) ? ((f29 > 2.5) ? ((f30 > 0.04542997) ? ((f6 > -0.1466458) ? ((f1 > 0.9351752) ? ((f23 > 0.9368812) ? ((f26 > 0.02921519) ? -0.00286918459164716 : -0.235087383104972) : ((f28 > 0.2766749) ? -0.178170197593943 : 0.105341438485046)) : ((f28 > 0.487019) ? 1.20571111635422 : -0.0342094989978545)) : 0.0854376966596912) : ((f16 > 60.5) ? ((f12 > 24.09407) ? -0.0507062433864029 : ((f9 > 5.9577) ? 0.0217720263873767 : ((f14 > 0.5828059) ? -0.062092577814657 : 0.090074636698314))) : ((f18 > 1.285872) ? ((f28 > 0.161677) ? -0.0535298397843367 : ((f1 > 0.6214421) ? 0.343378313603527 : 0.141423986148777)) : 0.0129469179171397))) : ((f30 > 0.01694018) ? -0.0186975715573832 : ((f31 > 3.5) ? ((f16 > 119.5) ? -0.036570149850259 : ((f4 > 0.01957069) ? ((f20 > 1.958988) ? ((f2 > 0.7819887) ? 0.293617525803497 : ((f2 > 0.3517269) ? 0.0201765281188038 : 1.06529937953718)) : ((f24 > 0.1009018) ? ((f32 > 0.1578004) ? ((f2 > 0.7128934) ? ((f27 > 0.2110251) ? -0.0701664587073034 : -0.292534799154888) : ((f1 > 0.3394569) ? -0.112879152777004 : ((f22 > 0.5743359) ? 0.101864562204699 : ((f26 > 0.4014906) ? 0.669682167929497 : 0.0894243772457632)))) : ((f3 > 0.3247496) ? ((f2 > 0.5723559) ? 0.186917412871738 : 0.0616217171090393) : ((f25 > 0.2376266) ? -0.0177447596662858 : ((f25 > 0.03775895) ? ((f24 > 0.1605776) ? 0.452036478252591 : 0.140735520736576) : -0.0311645351151039)))) : ((f23 > 0.1460426) ? -0.0491290654288034 : ((f25 > 0.3009301) ? ((f32 > 0.03496219) ? 0.132789770974685 : ((f27 > 0.155305) ? 1.10043188034237 : 0.227901606847084)) : -0.0194512502394317)))) : 0.144067721727439)) : -0.00319893132609288))) : -0.014736885451348);
            double treeOutput84 = ((f16 > 75.5) ? -0.0132110169664915 : ((f31 > 6.5) ? ((f32 > 0.07367785) ? ((f6 > -0.1044385) ? ((f28 > 0.02448963) ? ((f13 > 5.932066) ? 0.0528659683187139 : -0.0227651609257559) : ((f14 > 0.2076149) ? 0.165978555322828 : ((f2 > 0.4071964) ? ((f23 > 0.09608118) ? ((f0 > 0.8709032) ? ((f2 > 0.7004806) ? ((f26 > 0.2762148) ? ((f23 > 1.952857) ? -0.499921016245542 : -0.0123470446205199) : -0.198861002640098) : ((f1 > 0.3031648) ? 0.162007078048871 : -0.125780842839285)) : ((f32 > 0.1668385) ? ((f19 > 2.532057) ? -0.257627241936511 : -0.0233303770464889) : -0.0765139630915229)) : 0.111452558373094) : 0.0718210525737796))) : ((f9 > 18.29306) ? -0.0212284423381264 : 0.120358001483436)) : ((f15 > 0.6063874) ? ((f25 > 0.3954515) ? ((f30 > 0.01074824) ? 0.151947545766098 : 0.467120256241724) : ((f1 > 0.387057) ? 0.214567024008559 : -0.000916211183350168)) : 0.0277839184940909)) : ((f21 > 0.2744072) ? ((f26 > 0.5170648) ? ((f30 > 0.06577675) ? -0.110333147330054 : ((f0 > 0.8220387) ? ((f23 > 0.6386102) ? 0.183031643335037 : 0.600491468057504) : -0.0498728358528594)) : ((f2 > 0.1444462) ? ((f6 > 0.7640513) ? -0.163930969861284 : -0.06060074350961) : ((f2 > 0.06605282) ? ((f26 > 0.1249782) ? ((f18 > 1.221413) ? 0.738828058659602 : 0.428964851760629) : 0.018636748955804) : -0.0112014695816492))) : ((f29 > 2.5) ? ((f26 > 0.3348064) ? ((f29 > 6.5) ? 0.239823026531928 : -0.0330184286021946) : ((f29 > 6.5) ? -0.209427061946268 : ((f10 > 16.66873) ? ((f2 > 0.9313065) ? -0.00819425192018566 : ((f27 > 0.01060572) ? 0.0938916250956034 : ((f1 > 1.203706) ? 1.07356521558763 : 0.30059962811522))) : 0.116754923452464))) : 0.00972076470374226))));
            double treeOutput85 = ((f29 > 4.5) ? ((f26 > 0.2131944) ? ((f1 > 1.275451) ? ((f27 > 0.02782674) ? ((f31 > 3.5) ? 0.0469840133959872 : ((f4 > 0.496999) ? 0.684668950506936 : -0.403978441349245)) : 0.192361059487498) : ((f30 > 0.07080285) ? -0.0599703347258221 : ((f32 > 0.08649219) ? 0.0533562490904884 : -0.0170140888235653))) : ((f18 > 6.742136) ? ((f30 > 0.1378544) ? -0.363303534818631 : 0.251917064312374) : 0.0952779674945409)) : ((f23 > 0.534751) ? ((f0 > 1.240697) ? ((f3 > 0.1866705) ? ((f2 > 0.2096379) ? 0.0208002762222547 : ((f31 > 8.5) ? -0.0244763932125726 : 0.428778205329252)) : ((f28 > 0.177101) ? -0.339751146722956 : -0.0599553040984714)) : ((f32 > 0.04650059) ? ((f23 > 0.9485085) ? ((f25 > 0.4122096) ? -0.171371329728907 : -0.0114766578575001) : ((f0 > 1.054605) ? ((f2 > 1.069078) ? -0.0981727378391355 : 0.0899651813623526) : ((f32 > 0.312164) ? 0.264253713513619 : ((f9 > 7.505301) ? -0.0307836798053773 : ((f12 > 24.09407) ? ((f31 > 15.5) ? -0.322334729516405 : -0.0974361045977747) : -0.0580619531340103))))) : ((f2 > 1.248231E-05) ? -0.0165974236919191 : 21.5085005037864))) : ((f31 > 8.5) ? ((f27 > 0.3528365) ? ((f1 > 0.9519409) ? 0.580327424867483 : ((f29 > 1.5) ? -0.00137186849993989 : 0.457128335443004)) : ((f17 > 3.888973) ? ((f3 > 0.3772435) ? 0.0230396451243207 : -0.318552193569001) : ((f24 > 0.03848008) ? ((f26 > 0.07719615) ? ((f0 > 0.600227) ? ((f23 > 0.2805192) ? 0.147534328328788 : -0.0626266609580969) : ((f23 > 0.4514873) ? -0.0554856107017071 : ((f3 > 0.810124) ? -0.37337434728651 : 0.0651479051802939))) : 0.330216747156687) : ((f20 > 2.195245) ? 0.873335633535386 : -0.0390769866902215)))) : -0.0014098481808656)));
            double treeOutput86 = ((f15 > 0.8868023) ? 0.0232886814240261 : ((f30 > 0.02221101) ? ((f32 > 0.087467) ? ((f7 > -0.1691514) ? ((f7 > 0.3577823) ? ((f10 > 11.90008) ? -0.368437855985424 : -0.00544866553934634) : ((f14 > 0.6370224) ? 0.190070876852696 : 0.0259950188921075)) : ((f13 > 0.9852951) ? ((f14 > 0.03883861) ? 0.0646583099537377 : ((f6 > -0.1280944) ? -0.0372306699621414 : 0.0584861240814916)) : -0.17329944641038)) : ((f15 > 0.1243009) ? ((f2 > 0.01013615) ? ((f16 > 66.5) ? ((f29 > 5.5) ? ((f18 > 1.992776) ? ((f3 > 0.2577425) ? ((f30 > 0.07080285) ? -0.137304293742665 : 0.0284497972816872) : 0.209968236933617) : -0.0827290213817069) : ((f3 > 0.4299013) ? -0.0949581049879788 : ((f28 > 0.01874237) ? -0.0912207852403697 : -0.00524073990066291))) : ((f22 > 0.1879753) ? -0.0607789646634084 : 0.0133916488149979)) : 0.913285441750163) : -0.186721249332838)) : ((f29 > 1.5) ? ((f14 > 0.110101) ? ((f16 > 229.5) ? -0.127631929351379 : 0.0149925045394873) : ((f25 > 0.6499294) ? ((f30 > 0.01219075) ? 0.0529076361951285 : ((f6 > 0.1667082) ? 0.275980332455146 : 0.121381920276031)) : ((f18 > 1.902291) ? ((f28 > 0.1263339) ? ((f23 > 1.137263) ? ((f24 > 0.9981315) ? 0.394640961900128 : -0.251453238120885) : ((f32 > 0.1738583) ? -0.674570807807048 : 0.104391170331051)) : 0.155841547859028) : 0.018414668895588))) : ((f30 > 0.01030548) ? -0.0484568792123355 : ((f6 > -0.1030164) ? ((f10 > 18.11803) ? ((f7 > -0.1910965) ? ((f21 > 0.9684893) ? -0.135589428849393 : ((f31 > 0.5) ? ((f32 > 0.02837125) ? 0.029161567230346 : ((f30 > 0.005985465) ? 0.170896244535708 : 0.383678479762339)) : 0.264130979520172)) : -0.00598301724425283) : 0.00239100117446705) : -0.12084385916604)))));
            double treeOutput87 = ((f15 > 0.2005514) ? ((f14 > 0.6861408) ? ((f9 > 18.41715) ? ((f10 > 0.01398623) ? -0.507337164874368 : ((f26 > 0.4721169) ? 0.534175518561138 : ((f1 > 0.7135326) ? ((f5 > 0.2165978) ? ((f25 > 0.7509996) ? 3.08762744647825 : -0.328762855387852) : 0.712459932814445) : ((f31 > 6.5) ? ((f0 > 0.6852824) ? 0.694608876012625 : ((f25 > 0.522891) ? 1.20375944220424 : 1.05412030631846)) : 0.379481902361474)))) : -0.159801847511712) : ((f9 > 18.41715) ? ((f3 > 0.5128921) ? ((f32 > 0.04547565) ? ((f2 > 0.7436558) ? ((f25 > 0.6969948) ? ((f5 > 0.2115829) ? ((f7 > -0.2708555) ? 0.220704155011256 : -0.657984319570336) : -0.074429958051715) : ((f3 > 0.5722817) ? -0.077132799191176 : -0.225053410432639)) : 0.00798705749193373) : ((f6 > -0.06719974) ? 0.435331058842909 : 0.0805465097267986)) : ((f6 > -0.05584227) ? ((f0 > 0.7018881) ? -0.0299040315383248 : ((f4 > 0.2568724) ? ((f31 > 7.5) ? 0.0800028150778531 : ((f11 > -4.79049E-08) ? 0.471348012061009 : -0.456402951768755)) : 0.170179280144246)) : -0.0288148119933056)) : ((f29 > 3.5) ? ((f30 > 0.03875254) ? ((f16 > 82.5) ? ((f1 > 0.9190375) ? 0.0191529475695995 : ((f32 > 0.148212) ? 0.0733012317487353 : -0.0844290192540528)) : 0.0352374119116978) : ((f16 > 313.5) ? ((f29 > 11.5) ? 0.111710428317579 : -0.0710429933444369) : ((f0 > 0.8565323) ? ((f29 > 6.5) ? 0.213605157966545 : ((f23 > 0.8011545) ? ((f0 > 1.626267) ? 0.208557465921614 : 0.0407476112017314) : 0.184447058677184)) : 0.0323788190003036))) : ((f16 > 67.5) ? -0.0121362479171842 : ((f26 > 0.46027) ? ((f25 > 0.6344197) ? 0.109096957126854 : -0.0354814657629672) : 0.0150038470054191))))) : -0.0478084397178248);
            double treeOutput88 = ((f6 > -0.2251) ? ((f30 > 0.03296186) ? ((f1 > 0.8465301) ? ((f31 > 4.5) ? ((f21 > 1.822459) ? ((f0 > 1.379084) ? ((f26 > 0.8502229) ? -0.146830640308605 : 0.128780868523863) : ((f23 > 0.5190998) ? ((f28 > 0.1208459) ? -0.0839480633044125 : -0.353017967523452) : 0.123673774309884)) : ((f28 > 0.2049196) ? ((f2 > 0.7004806) ? ((f0 > 1.572526) ? ((f12 > 0.9221007) ? 0.302632443654526 : -0.324774417602007) : -0.176001872716303) : ((f2 > 0.5608138) ? 0.18707328391435 : -0.0279080778515564)) : ((f23 > 0.8829889) ? ((f29 > 9.5) ? 0.129996935980413 : ((f16 > 216.5) ? -0.266868905149931 : 0.0128265627807627)) : ((f3 > 0.4422233) ? ((f0 > 0.7960954) ? 0.127223084020678 : -0.0562248170163723) : ((f21 > 0.8857663) ? ((f23 > 0.3932374) ? 0.258746082574218 : 0.834705794766943) : 0.137141011848986))))) : -0.0582094035090288) : ((f27 > 0.5500337) ? ((f11 > -9.313226E-09) ? 0.24877240682374 : 0.996733706702651) : ((f29 > 1.5) ? ((f6 > 0.3730007) ? -0.0611246179815782 : -0.0201306923667152) : ((f27 > 0.02706412) ? 0.515644606351061 : -0.103550941121518)))) : ((f29 > 2.5) ? ((f23 > 0.1665278) ? ((f16 > 236.5) ? -0.023481034645704 : ((f1 > 0.4171919) ? ((f15 > 0.3448164) ? ((f29 > 5.5) ? 0.279657970770773 : ((f16 > 106.5) ? ((f30 > 0.01959911) ? -0.00802148970566439 : 0.19773650433563) : ((f28 > 0.1972347) ? -0.100956703221592 : 0.160844000740642))) : ((f29 > 6.5) ? 0.13913360664728 : 0.0154766607337779)) : ((f19 > 2.184967) ? 0.109232847453741 : ((f3 > 0.2295256) ? -0.0487178270805924 : 0.0631410826295124)))) : ((f27 > 0.4001538) ? 0.516903066589465 : -0.159038873812452)) : -0.000252666027058048)) : -0.127168559963589);
            double treeOutput89 = ((f15 > 0.6063874) ? ((f31 > 10.5) ? ((f27 > 0.001190391) ? ((f29 > 2.5) ? ((f5 > 0.0649479) ? ((f23 > 0.3657564) ? ((f25 > 1.450612) ? -0.376115781195365 : 0.0395554659578089) : -0.113341936658875) : ((f4 > 0.3688505) ? ((f29 > 7.5) ? -0.246193149638234 : ((f16 > 269.5) ? 0.560119005181052 : 0.0113997912306613)) : ((f2 > 0.09828465) ? ((f25 > 0.2961216) ? ((f12 > 28.72018) ? -0.126116201172117 : 0.100928652051408) : ((f1 > 0.3210928) ? ((f1 > 0.6860125) ? 0.330457680008853 : 0.48833543907668) : 0.17469778106208)) : -0.0846862203167825))) : ((f32 > 0.1399797) ? ((f25 > 0.8051723) ? 0.213757722176578 : ((f28 > 0.2702795) ? 0.098762849166603 : -0.0985150158074569)) : ((f26 > 0.2895273) ? ((f23 > 0.4305942) ? 0.0187550116169673 : 0.163995580580957) : -0.0229685049877209))) : ((f26 > 0.6236803) ? 0.269499548410407 : ((f2 > 1.155865) ? -0.378257098533064 : ((f32 > 0.1110331) ? 0.1443214391271 : -0.193100098805526)))) : 0.00502597592556448) : ((f30 > 0.01886052) ? ((f32 > 0.1052382) ? 0.0125713687159722 : ((f19 > 2.127961) ? ((f0 > 1.313963) ? -0.012995736859205 : ((f23 > 0.7334907) ? ((f21 > 1.151615) ? -0.245171704020578 : -0.0925239547906988) : -0.0418008937715733)) : ((f31 > 22.5) ? 0.136748756199873 : ((f21 > 0.1183571) ? ((f24 > 0.204077) ? ((f26 > 0.5170648) ? -0.233263757291803 : ((f24 > 0.9981315) ? ((f25 > 0.5134943) ? 0.62757101384468 : 0.20698168215475) : -0.0256976502966861)) : -0.112297185320507) : ((f29 > 2.5) ? ((f25 > 0.3009301) ? 0.0287081217821053 : ((f25 > 0.08818689) ? ((f32 > 0.04515826) ? 0.413901679390645 : 0.220233798893512) : -0.055871345938396)) : -0.0694734894728464))))) : -0.00138743370025128));
            double treeOutput90 = ((f16 > 113.5) ? -0.0235411938803793 : ((f29 > 4.5) ? ((f9 > 15.68326) ? -0.0299940579260869 : ((f30 > 0.09833287) ? -0.0202480524389582 : ((f18 > 2.434362) ? ((f30 > 0.04080816) ? ((f15 > 0.1291373) ? 0.0906916129260922 : -0.209012889841954) : 0.174259281896032) : 0.0171808601142455))) : ((f30 > 0.0425678) ? -0.0257803308690494 : ((f1 > 0.03150564) ? ((f2 > 0.2235942) ? ((f25 > 0.6552966) ? ((f4 > 0.4179891) ? 0.153190814420929 : 0.0184647653466844) : ((f3 > 0.676752) ? -0.120526747628509 : ((f18 > 2.467326) ? ((f12 > 0.9221007) ? ((f0 > 0.3233415) ? ((f29 > 1.5) ? 0.20304980069664 : 0.034807014368183) : -0.142746794895619) : -0.20899364290076) : ((f28 > 0.4203187) ? 0.827093991721176 : -0.0107392916406611)))) : ((f25 > 0.04537791) ? ((f28 > 0.02739833) ? ((f24 > 0.1037236) ? ((f30 > 0.01753577) ? ((f26 > 0.1600224) ? 0.114783237961592 : -0.152003888366457) : ((f15 > 0.3923042) ? ((f1 > 0.2725219) ? 0.516065975409311 : 0.248584635147728) : 0.131272390267058)) : ((f30 > 0.003847634) ? -0.290954019252621 : 0.27967436451791)) : ((f31 > 6.5) ? ((f2 > 0.08630696) ? ((f26 > 0.1577641) ? 0.47665021288322 : 0.191210197813964) : 0.0414038538833474) : 0.119125582021893)) : ((f23 > 0.2197619) ? ((f5 > 0.005724818) ? ((f23 > 0.382246) ? 0.0415543154412364 : 0.393813341806626) : -0.0966665705369389) : ((f28 > 0.005668942) ? 0.660864816616244 : 0.468077060036505)))) : ((f32 > 0.1052382) ? -0.158131488750002 : ((f24 > 0.002070132) ? ((f2 > 0.05221485) ? -0.169807442047727 : 0.203457649191467) : ((f17 > 0.5440315) ? 0.100684031973626 : ((f25 > 0.2812213) ? ((f3 > 0.3742794) ? -0.0723019235761051 : 0.508810177455752) : -0.0968264864617883))))))));
            double treeOutput91 = ((f18 > 7.860124) ? ((f32 > 0.06251879) ? ((f12 > 0.6421817) ? 0.0579833173218714 : 0.324877425813818) : 0.39534747080635) : ((f7 > -0.4084556) ? ((f15 > 0.08699) ? ((f30 > 0.05493618) ? ((f1 > 0.2511762) ? ((f32 > 0.1668385) ? ((f28 > 0.2049196) ? ((f26 > 0.4785053) ? 0.159261412855861 : ((f29 > 4.5) ? -0.252298856729564 : 0.11234144680587)) : ((f3 > 0.3998359) ? ((f3 > 0.6209276) ? ((f17 > 4.836257) ? 0.182106970227822 : ((f2 > 1.09351) ? 0.200686134373022 : ((f27 > 0.1871063) ? ((f30 > 0.1270167) ? -0.216777824162968 : 0.269347933121133) : ((f0 > 0.9415609) ? ((f3 > 0.8832791) ? ((f24 > 0.5794598) ? -0.0231153085198509 : -0.433460352715323) : 0.181835692317578) : ((f27 > 0.07560255) ? -0.95456506272733 : -0.035345995038017))))) : ((f32 > 0.2107053) ? 0.310016451350893 : 0.132429896280736)) : -0.0405322709102163)) : ((f21 > 2.09486) ? ((f5 > 4.309524E-05) ? -0.00881107076502841 : -0.359382384246399) : -0.0277010086127167)) : ((f16 > 166.5) ? 0.68527626228119 : 0.177905654417426)) : ((f29 > 4.5) ? ((f16 > 139.5) ? ((f29 > 7.5) ? 0.0519813631622552 : -0.0319510034899446) : ((f24 > 0.3402435) ? 0.0924145864212911 : -0.00701964020761209)) : ((f16 > 57.5) ? ((f30 > 0.02683053) ? ((f23 > 0.6214191) ? ((f31 > 2.5) ? ((f0 > 1.52381) ? 0.026100449304772 : -0.0634642921639556) : ((f32 > 0.1363017) ? 0.580548413743641 : 0.0980954150822217)) : 0.00120232552130749) : 0.00177296256504201) : ((f26 > 0.46027) ? ((f25 > 0.7255468) ? ((f15 > 0.5684311) ? 0.24105893457519 : 0.0644876561921047) : 0.0130608138919359) : ((f0 > 2.208209) ? -0.215943765935689 : 0.0243447581442439))))) : -0.108603967307261) : -0.0430484415122337));
            double treeOutput92 = ((f16 > 49.5) ? ((f15 > 0.7721616) ? ((f1 > 0.5470313) ? ((f30 > 0.05126864) ? 0.000144281707557003 : ((f23 > 0.7334907) ? 0.0212030800167685 : ((f23 > 0.2103111) ? 0.10590651475232 : -0.0130514173260571))) : 0.00556345157858433) : ((f30 > 0.01886052) ? ((f32 > 0.08197668) ? 0.00330940974527933 : ((f26 > 0.3691205) ? -0.0546657186787996 : -0.0207057096334686)) : ((f31 > 7.5) ? ((f2 > 0.09407881) ? ((f2 > 0.2377282) ? ((f20 > 1.441547) ? ((f18 > 1.142924) ? ((f29 > 2.5) ? ((f14 > 0.09918033) ? 0.0569048423022233 : 0.196669470169297) : ((f25 > 0.7106913) ? ((f24 > 0.6322328) ? -0.165283424921036 : ((f3 > 0.5077426) ? 0.310908162830142 : 0.0459587057163553)) : ((f9 > 2.816224) ? -0.0895194718496524 : -0.309823953962523))) : ((f26 > 0.4852667) ? ((f31 > 12.5) ? 0.427837245804991 : 0.189263035267397) : 0.102652760887952)) : ((f32 > 0.03702501) ? ((f13 > 14.07972) ? -0.067613156469924 : ((f11 > 3.742753E-08) ? -0.0451871431448861 : 0.0332568093035236)) : ((f6 > 0.0375234) ? ((f16 > 162.5) ? -0.0162285270409058 : ((f15 > 0.2292333) ? ((f18 > 0.934253) ? 0.512080585385749 : 0.194752073772404) : 0.132108716118442)) : -0.064642568585458))) : ((f18 > 0.5415608) ? ((f32 > 0.06428925) ? 0.499099431899896 : ((f31 > 14.5) ? 0.366954292936508 : 0.194837145388111)) : 0.0347085620309875)) : ((f18 > 2.812126) ? ((f26 > 0.2028701) ? -0.132530933743481 : ((f4 > 0.1978723) ? -0.68543198501569 : -0.037879632564604)) : ((f5 > 0.05748396) ? ((f23 > 0.3269992) ? ((f32 > 0.06850072) ? 0.446930937190169 : 0.0650031855359232) : 1.01863439143995) : -0.153653223921278))) : -0.0156407979097956))) : ((f5 > 0.00509081) ? 0.850153162626814 : 0.345015844889604));
            double treeOutput93 = ((f31 > 47.5) ? 0.171122822336615 : ((f16 > 188.5) ? ((f11 > 5.587935E-08) ? ((f26 > 0.2131944) ? -0.107705227922557 : -0.0699943319900948) : ((f21 > 1.007897) ? ((f16 > 1353.5) ? ((f1 > 0.3821421) ? 0.421971256810158 : 1.5494706548834) : ((f10 > 11.51293) ? 0.219792861092527 : 0.032434919886895)) : ((f25 > 1.258936) ? ((f24 > 0.192377) ? 0.165994694567643 : 0.69600131554793) : ((f23 > 1.631086) ? -0.233925789818541 : ((f29 > 9.5) ? ((f14 > 0.3485162) ? -0.077303608046921 : ((f23 > 1.239685) ? ((f18 > 3.486244) ? 0.253802781082888 : ((f3 > 0.4784039) ? ((f5 > 0.1932853) ? 1.077362252608 : 0.0375921107066701) : ((f16 > 762.5) ? -0.715803004722315 : -0.160779411788358))) : 0.218102748130134)) : ((f1 > 0.9351752) ? -0.162251318037609 : -0.0125293308708822)))))) : ((f29 > 18.5) ? -0.195035592972392 : ((f31 > 20.5) ? ((f10 > 1.796722) ? ((f27 > 0.152519) ? ((f30 > 0.02369434) ? 0.116287503953663 : 0.195489121047422) : ((f32 > 0.0545509) ? ((f8 > 0.9832786) ? ((f27 > 0.03495352) ? 0.0889318482713142 : -0.247614829158574) : ((f29 > 5.5) ? ((f4 > 0.02848527) ? 0.237300428360909 : -0.524467780807022) : ((f2 > 0.7004806) ? ((f26 > 0.372528) ? -0.192024817701033 : -0.731444917173716) : -0.13913244399041))) : 0.167108592921343)) : -0.0671205867365164) : ((f15 > 0.1525831) ? ((f11 > 1.933807) ? ((f10 > 9.904465) ? 0.00253132455774139 : ((f31 > 3.5) ? ((f14 > 0.035401) ? ((f16 > 93.5) ? 0.0883633612288711 : ((f11 > 13.74151) ? 0.462938588666366 : 0.197509224177371)) : 0.0420466842549675) : 0.0610620217752636)) : ((f24 > 1.228056) ? 0.0848707630805802 : -0.00157385209250682)) : -0.0650985035677468)))));
            double treeOutput94 = ((f4 > 0.1096569) ? ((f26 > 0.2467872) ? ((f16 > 127.5) ? ((f31 > 38.5) ? 0.151837477860945 : -0.0486193989674292) : 0.0132360505615944) : ((f19 > 1.843888) ? -0.0995740551071211 : ((f24 > 0.2099921) ? ((f14 > 0.04396135) ? 0.100277674906143 : ((f5 > -0.0001097834) ? ((f28 > 0.08534105) ? ((f2 > 0.401704) ? 0.18393840938507 : -0.134215258232115) : 0.0499853458267463) : -0.248651569348069)) : ((f28 > 0.1856125) ? ((f4 > 0.2035397) ? 0.1284199084189 : 1.10332064895983) : -0.108854392478032)))) : ((f4 > 1.519449E-05) ? ((f27 > 0.01463856) ? ((f24 > 0.0670104) ? ((f23 > 0.3710835) ? ((f10 > 18.42068) ? ((f2 > 0.2235942) ? ((f6 > -0.07182601) ? ((f3 > 0.676752) ? 0.064037044391755 : -0.0910087490733393) : 0.0123198755029607) : 0.0604086337457528) : 0.0155313173668881) : ((f0 > 0.8933967) ? -0.167752862504765 : ((f2 > 0.7004806) ? ((f5 > 0.01699469) ? ((f24 > 0.3031055) ? -0.112613570309263 : 0.306976573398632) : 0.233913677457614) : ((f32 > 0.08473016) ? 0.196288783889404 : ((f26 > 0.3036482) ? -0.140714035945231 : 0.0781627026348472))))) : ((f5 > 0.1362104) ? 1.86333473723933 : ((f2 > 0.06837622) ? -0.0716673365872595 : 0.110116855731711))) : ((f28 > 0.0175096) ? 0.412291407616123 : 0.0925469257435091)) : ((f19 > 2.224677) ? ((f0 > 1.477073) ? 0.0161347738119507 : -0.0553514794585292) : ((f31 > 3.5) ? ((f18 > 2.572657) ? ((f32 > 0.1363017) ? 0.667422156939791 : ((f23 > 1.02504) ? 0.0866696722953553 : 0.420162230183102)) : ((f26 > 0.1860681) ? -0.0147570930587244 : ((f32 > 0.04082602) ? ((f30 > 0.01379628) ? ((f25 > 0.08818689) ? 0.214515439726246 : -0.215708319282058) : 0.491143307650535) : 0.138856451997221))) : -0.0291194050282567))));
            double treeOutput95 = ((f15 > 0.5159602) ? ((f29 > 3.5) ? ((f30 > 0.06450012) ? ((f31 > 2.5) ? 0.000172728710005108 : ((f25 > 1.258936) ? 0.464868469666349 : -0.262403953885405)) : ((f16 > 640.5) ? ((f10 > 16.49036) ? 0.314913151813444 : -0.103266214292663) : ((f29 > 5.5) ? ((f32 > 0.259126) ? -0.146807339123709 : 0.101147784659144) : ((f16 > 83.5) ? ((f30 > 0.03772757) ? -0.0675333018223445 : ((f16 > 128.5) ? ((f30 > 0.0212679) ? -0.0472109846132922 : 0.0917222644847422) : ((f26 > 0.4721169) ? 0.284030431884454 : 0.0643185472320718))) : ((f31 > 4.5) ? ((f13 > 4.885828) ? 0.0828587703890383 : -0.00878571487295996) : ((f0 > 1.014758) ? ((f31 > 3.5) ? 0.172614298146553 : 0.507945395501588) : 0.0590621349285824)))))) : ((f30 > 0.02920031) ? -0.0184470280435374 : ((f29 > 1.5) ? ((f16 > 69.5) ? ((f30 > 0.01961497) ? -0.0183725721154489 : ((f16 > 194.5) ? -0.0679207042518809 : ((f30 > 0.008337195) ? 0.0599832634751801 : ((f18 > 1.335444) ? 1.26891200695797 : 0.446708034802927)))) : ((f26 > 0.4785053) ? ((f0 > 0.562814) ? 0.34684235756568 : -0.0256237029234516) : ((f18 > 1.79629) ? ((f32 > 0.1935912) ? -0.190545597495641 : ((f15 > 0.7619301) ? ((f10 > 0.8284973) ? 0.387102963305103 : -0.140708128479536) : ((f30 > 0.02231906) ? 0.0949661561238164 : 0.420054601517489))) : 0.0374295721228026))) : ((f30 > 0.01724846) ? ((f23 > 0.24872) ? -0.110586207271909 : ((f27 > 0.1459137) ? -0.207731352286694 : ((f25 > 0.3289494) ? ((f32 > 0.123332) ? -0.333215758461605 : ((f17 > 1.630582) ? 0.0187504513391613 : 0.192618367582572)) : -0.104271698764055))) : 0.00913870493018133)))) : ((f30 > 0.01961497) ? -0.022178628785051 : -0.00562791180871185));
            double treeOutput96 = ((f16 > 87.5) ? ((f9 > 2.666536) ? ((f29 > 6.5) ? ((f30 > 0.09474178) ? -0.0684637825058577 : ((f18 > 3.636645) ? ((f27 > 0.01127567) ? 0.0776870003572791 : ((f2 > 0.8629014) ? 0.0490806544975901 : 0.594164912790892)) : ((f28 > 0.1335869) ? ((f20 > 1.527422) ? ((f0 > 1.255128) ? 0.289917484213535 : -0.045226533981947) : ((f0 > 1.255128) ? -0.232932874418167 : -0.0466281186259796)) : ((f5 > 4.309524E-05) ? ((f1 > 0.7230155) ? 0.221371307684875 : ((f23 > 0.4617791) ? 0.0756076952495835 : ((f26 > 0.7179694) ? 1.27941704606635 : -0.214647868245051))) : ((f31 > 10.5) ? ((f32 > 0.1668385) ? 0.473137016873375 : ((f27 > 0.02937504) ? -0.0288730197960822 : ((f2 > 0.6405799) ? -0.29515349410047 : ((f1 > 0.540709) ? 0.451966783966489 : -0.229008481154874)))) : ((f3 > 0.3897421) ? ((f32 > 0.08569203) ? 0.230174913654202 : 0.0669746047338449) : -0.060637995055637)))))) : ((f23 > 0.6046541) ? ((f2 > 1.248231E-05) ? ((f30 > 0.03007862) ? -0.0648904589861649 : -0.0344478174109804) : -8.79369575908785) : ((f25 > 0.7706231) ? ((f31 > 2.5) ? ((f1 > 0.5284226) ? ((f5 > 0.02031295) ? -0.125351240320079 : ((f9 > 8.34276) ? -0.0198314418976019 : 0.217844543663639)) : ((f16 > 215.5) ? ((f9 > 18.03865) ? 1.40604797317196 : 0.474626254802383) : 0.174066595378746)) : -0.225069787022716) : -0.0215562779110424))) : 0.0479038898048361) : ((f11 > 0.7028403) ? ((f18 > 1.696045) ? ((f21 > 0.01770619) ? ((f27 > 0.2381288) ? -0.0147019901201971 : ((f25 > 0.05623567) ? ((f15 > 0.6513382) ? 0.163129682783891 : 0.0967600012417635) : -0.0503814207951702)) : -0.0232939131960886) : ((f13 > 0.02541088) ? 0.0202552941811778 : 0.56486638525352)) : 0.000960729131371223));
            double treeOutput97 = ((f1 > 1.027983) ? ((f26 > 0.795416) ? ((f0 > 0.8088951) ? -0.0218473926265212 : ((f1 > 1.49885) ? -0.0327017367578021 : -0.392130852919372)) : ((f9 > 15.04411) ? -0.0343136414108354 : ((f2 > 1.561375) ? ((f5 > 0.01653337) ? -0.245272260080672 : -0.0121803789565832) : ((f31 > 3.5) ? ((f7 > 0.9524139) ? -0.0701107690250395 : 0.0830366622495986) : ((f4 > 0.4441568) ? 0.361647601336905 : ((f4 > 0.2217676) ? ((f32 > 0.05404603) ? -0.396703281045401 : -0.220997166102648) : 0.0127728122818942)))))) : ((f26 > 1.080976) ? 0.407755949012225 : ((f27 > 0.5500337) ? ((f0 > 1.926173) ? 0.708476257293597 : ((f23 > 1.313321) ? -0.301194723287012 : 0.249322860801966)) : ((f30 > 0.02880905) ? ((f6 > 0.2736513) ? ((f23 > 0.8539834) ? ((f26 > 0.2654634) ? ((f32 > 0.1399797) ? -0.0111638909462458 : ((f14 > 0.3066239) ? -0.0406551090285727 : -0.155364983926656)) : 0.0176808414533958) : -0.0201218369901898) : ((f9 > 2.98332) ? -0.00318777509492659 : -0.120624168804004)) : ((f31 > 6.5) ? ((f19 > 3.415355) ? ((f3 > 0.5994364) ? 0.305555886283893 : 0.0339490042961724) : ((f13 > 10.12137) ? ((f32 > 0.1052382) ? ((f21 > 0.2948012) ? -0.0762299974587955 : 0.132620868671209) : -0.00328788682869779) : ((f12 > 9.129647) ? ((f14 > 0.01612903) ? ((f11 > 4.284084E-08) ? ((f11 > 2.525729) ? 0.111728051052136 : ((f9 > 3.536758) ? -0.116428639765731 : 0.0671683864739212)) : 0.162280003098125) : ((f13 > 9.897587) ? 0.149286097921793 : -0.0850202611331804)) : ((f25 > 0.4690152) ? ((f24 > 0.3324618) ? ((f3 > 0.3541545) ? ((f2 > 0.7819887) ? 0.142923256403546 : -0.0195112624212539) : -0.109297114875155) : 0.1162650621904) : 0.0296916580605349)))) : -0.0103947801868911)))));
            double treeOutput98 = ((f24 > 1.733555) ? 0.120764972195953 : ((f23 > 1.631086) ? ((f0 > 2.208209) ? ((f30 > 0.07080285) ? ((f22 > 0.3910032) ? 0.19401068063573 : ((f4 > 0.5830308) ? ((f11 > 2.980232E-08) ? -0.611523618242214 : -0.171171616513258) : ((f25 > 0.9742545) ? 0.157199502031046 : ((f32 > 0.1578004) ? 0.251480020766236 : -0.496857376128345)))) : ((f26 > 0.3976818) ? ((f2 > 1.819301) ? -0.250255949504257 : 0.286032131234643) : -0.0519225650523353)) : ((f14 > 0.03960785) ? ((f28 > 0.3823924) ? 0.429441779780924 : -0.0566603840860874) : ((f0 > 1.858214) ? ((f23 > 1.747192) ? -0.167319796023218 : -0.00788628750566213) : ((f29 > 4.5) ? -0.151699689523711 : ((f3 > 0.4784039) ? ((f5 > 0.004406244) ? ((f27 > 0.1012524) ? -0.382568363036259 : -1.20317672526594) : -0.428100867948205) : -0.217527359482905))))) : ((f22 > 2.426597) ? -0.376967833361155 : ((f24 > 1.091105) ? ((f5 > 0.5502803) ? ((f25 > 0.9464693) ? -0.757678195396568 : -0.158211815756827) : 0.0804147477523779) : ((f17 > 9.248114) ? ((f26 > 0.6653943) ? 0.437830640774134 : -0.1994052587623) : ((f15 > 0.2960026) ? ((f12 > 3.893767) ? ((f13 > 4.885828) ? ((f13 > 8.127401) ? 0.00625308390316571 : ((f12 > 8.229173) ? -0.00351573266587008 : ((f25 > 0.4530378) ? 0.0883177623987357 : 0.0710069086216074))) : ((f12 > 4.111117) ? -0.0364628629415383 : 0.0936599663297869)) : ((f13 > 3.135156) ? -0.0575644921314722 : ((f7 > 0.4202187) ? ((f6 > 0.09471093) ? -0.274583405589826 : ((f7 > 0.9878271) ? 0.210483638129936 : -0.111835624786198)) : ((f12 > 3.118154) ? -0.0948131507120676 : ((f31 > 4.5) ? 0.0562685426290024 : 0.0115408306927382))))) : ((f30 > 0.0217307) ? -0.049781142378585 : -0.0200475135370004)))))));
            double treeOutput99 = ((f29 > 4.5) ? ((f28 > 0.09136954) ? ((f23 > 0.5882032) ? 0.00520358067490664 : ((f25 > 0.166528) ? ((f25 > 0.2812213) ? ((f28 > 0.3247706) ? -0.377718193077482 : ((f21 > 1.10939) ? ((f26 > 0.3238384) ? 0.116628389003838 : 0.85782863773726) : -0.174914740935598)) : -0.532370769818744) : 0.306122591310184)) : ((f5 > 4.309524E-05) ? ((f24 > 0.216031) ? ((f26 > 0.2988414) ? 0.0532610794446854 : ((f23 > 0.8181025) ? ((f25 > 1.450612) ? -0.256557076657658 : 0.285023975637734) : ((f25 > 0.595188) ? -0.142736506702495 : ((f18 > 1.946636) ? 0.325102664677112 : 0.101792969021013)))) : -0.0702243900254328) : ((f23 > 0.493018) ? ((f0 > 0.8933967) ? ((f23 > 0.7622578) ? 0.00117320914133544 : 0.265026481810125) : ((f21 > 1.09047) ? -0.412633870600686 : -0.0572715250574813)) : 0.107456434492434))) : ((f32 > 0.04515826) ? ((f6 > 0.5739934) ? ((f5 > 0.1850025) ? 0.0603025268627802 : ((f24 > 0.1953017) ? -0.0309815249239882 : ((f31 > 13.5) ? 0.0290542212082336 : -0.122499932260308))) : ((f9 > 2.902979) ? ((f12 > 25.47787) ? ((f2 > 0.5781242) ? ((f31 > 17.5) ? ((f21 > 0.6823651) ? 0.0989452002030045 : ((f30 > 0.01162228) ? -0.63581772157793 : -0.244472052687934)) : -0.124947657744698) : -0.000243070723539685) : -0.00151721347943946) : ((f14 > 0.3036891) ? 0.0906985367630706 : -0.0784110277009221))) : ((f9 > 2.48424) ? ((f5 > 0.003619287) ? ((f28 > 0.01832819) ? 0.0175745709274828 : ((f1 > 0.677354) ? 0.320609729225832 : ((f24 > 0.02660193) ? ((f29 > 0.5) ? 0.132832836478251 : 0.430363517442865) : -0.0590681871451304))) : ((f7 > 0.9981551) ? 0.178762501059981 : ((f10 > 17.7122) ? 0.0442902971028191 : -0.0485125349385178))) : 0.124069343302725)));
            double output = treeOutput0 + treeOutput1 + treeOutput2 + treeOutput3 + treeOutput4 + treeOutput5 + treeOutput6 + treeOutput7 + treeOutput8 + treeOutput9 + treeOutput10 + treeOutput11 + treeOutput12 + treeOutput13 + treeOutput14 + treeOutput15 + treeOutput16 + treeOutput17 + treeOutput18 + treeOutput19 + treeOutput20 + treeOutput21 + treeOutput22 + treeOutput23 + treeOutput24 + treeOutput25 + treeOutput26 + treeOutput27 + treeOutput28 + treeOutput29 + treeOutput30 + treeOutput31 + treeOutput32 + treeOutput33 + treeOutput34 + treeOutput35 + treeOutput36 + treeOutput37 + treeOutput38 + treeOutput39 + treeOutput40 + treeOutput41 + treeOutput42 + treeOutput43 + treeOutput44 + treeOutput45 + treeOutput46 + treeOutput47 + treeOutput48 + treeOutput49 + treeOutput50 + treeOutput51 + treeOutput52 + treeOutput53 + treeOutput54 + treeOutput55 + treeOutput56 + treeOutput57 + treeOutput58 + treeOutput59 + treeOutput60 + treeOutput61 + treeOutput62 + treeOutput63 + treeOutput64 + treeOutput65 + treeOutput66 + treeOutput67 + treeOutput68 + treeOutput69 + treeOutput70 + treeOutput71 + treeOutput72 + treeOutput73 + treeOutput74 + treeOutput75 + treeOutput76 + treeOutput77 + treeOutput78 + treeOutput79 + treeOutput80 + treeOutput81 + treeOutput82 + treeOutput83 + treeOutput84 + treeOutput85 + treeOutput86 + treeOutput87 + treeOutput88 + treeOutput89 + treeOutput90 + treeOutput91 + treeOutput92 + treeOutput93 + treeOutput94 + treeOutput95 + treeOutput96 + treeOutput97 + treeOutput98 + treeOutput99;
            return output;
        }
        #endregion
    }
}
