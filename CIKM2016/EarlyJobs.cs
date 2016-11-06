using CIKM2016.Models;
using CIKM2016.Structure;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIKM2016
{
    class EarlyJobs
    {

        public static void UrlComAnalysis()
        {
            var fid2url = Loader.LoadFid2Url();
            string gtfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\common_user_stat\url02.csv";
            var user2matches = Loader.LoadGroundTruth(gtfile);
            var user2fact = Loader.LoadUserFacts(user2matches);
            var fact2url = Loader.LoadFid2Url();

            Dictionary<string, HashSet<string>> user2urls = new Dictionary<string, HashSet<string>>();

            int target_dep = 1;
            HashSet<string> uid_set = new HashSet<string>(user2matches.Keys);
            user2matches.Clear();
            user2matches = null;

            foreach (var uid in uid_set)
            {
                user2urls.Add(uid, new HashSet<string>());
                foreach (var fact in user2fact[uid].facts)
                {
                    string target_url = ExtractUrlWithDep(target_dep, fact2url[fact.fid]);
                    if (target_url == null)
                    {
                        continue;
                    }
                    if (!user2urls[uid].Contains(target_url))
                    {
                        user2urls[uid].Add(target_url);
                    }
                }
            }

           
            Dictionary<string, int> url02cnt = new Dictionary<string, int>();
            using(StreamReader rd = new StreamReader(gtfile)) 
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    if (user2urls.ContainsKey(words[0]) && user2urls.ContainsKey(words[1]))
                    {
                        foreach (var url in user2urls[words[0]])
                        {
                            if (user2urls[words[1]].Contains(url))
                            {
                                if (!url02cnt.ContainsKey(url))
                                {
                                    url02cnt.Add(url, 1);
                                }
                                else
                                {
                                    url02cnt[url]++;
                                }
                            }
                        }
                    }
                }
            }

            Naive.OutputSortedDict(url02cnt, outfile);

        }
        public static void UrlComAnalysis_random()
        {
            var fid2url = Loader.LoadFid2Url(); 
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\common_user_stat\url_random_2w.csv";  

            var fact2url = Loader.LoadFid2Url(1);
            var user2facts = Loader.LoadUserFacts();

            Dictionary<string, HashSet<string>> user2urls = new Dictionary<string, HashSet<string>>();

            List<string> uid_set = new List<string>(user2facts.Keys);

            foreach (var uid in uid_set)
            {
                var cset = new HashSet<string>();
                foreach (var fact in user2facts[uid].facts)
                {
                    if (fact2url.ContainsKey(fact.fid))
                    {
                        if (!cset.Contains(fact2url[fact.fid]))
                        {
                            cset.Add(fact2url[fact.fid]);
                        }
                    }
                }
                user2urls.Add(uid, cset);
            }


            Dictionary<string, int> url02cnt = new Dictionary<string, int>();

            long N = 33000 * 20000;
            int len = uid_set.Count;
            Random rng = new Random((int)DateTime.Now.Ticks);
            for (long i = 0; i < N; i++)
            {
                if (i % 10000L == 0)
                {
                    Console.WriteLine(i);
                }
                int a = rng.Next(len);
                int b = rng.Next(len);
                if (a == b)
                {
                    continue;
                }
                string[] words = new string[] { uid_set[a], uid_set[b] };

                if (user2urls.ContainsKey(words[0]) && user2urls.ContainsKey(words[1]))
                {
                    foreach (var url in user2urls[words[0]])
                    {
                        if (user2urls[words[1]].Contains(url))
                        {
                            if (!url02cnt.ContainsKey(url))
                            {
                                url02cnt.Add(url, 1);
                            }
                            else
                            {
                                url02cnt[url]++;
                            }
                        }
                    }
                }

            }

            Naive.OutputSortedDict(url02cnt, outfile);

        }

        public static void ValidDetection()
        {
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");
            string infile = @"D:\tmp\user-linking\ls-65-input\submission.txt";
            int hit = 0;
            HashSet<string> visited = new HashSet<string>();
            using (StreamReader rd = new StreamReader(infile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    if (words[0].CompareTo(words[1]) >= 0)
                    {
                        hit++;
                        continue;
                    }
                    if (!visited.Contains(content))
                    {
                        visited.Add(content);
                    }
                    else
                    {
                        hit++;
                        continue;
                    }
                    if (user2matches.ContainsKey(words[0]) || user2matches.ContainsKey(words[1]))
                    {
                        hit++;
                        continue;
                    }
                }
            }
            Console.WriteLine("error {0} ", hit);
        }

        public static string ExtractUrlWithDep(int target_dep, List<string> list)
        {
            if (list.Count > target_dep)
            {
                return list[target_dep];
            }
            else
            {
                return null;
            }
        }

        public static void StatUserFidCnt()
        {
            Dictionary<string, int> user2factcnt = new Dictionary<string, int>();
            int factcnt = 0;   

            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\facts.json";
            using (StreamReader rd = new StreamReader(infile))
            {
                factcnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (factcnt++ % 10000 == 0)
                    {
                        Console.WriteLine(factcnt);
                    }

                    Facts ss = JsonConvert.DeserializeObject<Facts>(content);
                     
                    if (!user2factcnt.ContainsKey(ss.uid))
                    {
                        user2factcnt.Add(ss.uid, ss.facts.Count);
                    }
                    else
                    {
                        user2factcnt[ss.uid] += ss.facts.Count;
                    }  
                }
            }

            
            Console.WriteLine("user cnt : {0}", user2factcnt.Count);
            Console.WriteLine("user max fact cnt : {0}", user2factcnt.Max(a => a.Value));

            Naive.OutputSortedDict(user2factcnt, @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\user2fidcnt.csv");
        }

        public static void Stat01()
        {
            Dictionary<string, int> user2factcnt = new Dictionary<string, int>();
            int factcnt = 0;
            int eventcnt = 0;
            List<double> facttime = new List<double>();
            DateTime mint = DateTime.Now;
            DateTime maxt = DateTime.Now.AddYears(-20);

            HashSet<string> fidset = new HashSet<string>();

            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\facts.json";
            using (StreamReader rd = new StreamReader(infile))
            {
                factcnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (factcnt++ % 10000 == 0)
                    {
                        Console.WriteLine(factcnt);
                    }

                    Facts ss = JsonConvert.DeserializeObject<Facts>(content);
                    DateTime t1 = DateTime.Now;
                    DateTime t2 = DateTime.Now.AddYears(-20);
                    if (!user2factcnt.ContainsKey(ss.uid))
                    {
                        user2factcnt.Add(ss.uid, 1);
                    }
                    else
                    {
                        user2factcnt[ss.uid]++;
                    }

                    
                    foreach (var re in ss.facts)
                    {
                         
                        var curt = UserProfileInfer.Utils.Common.ParseTimeStampMillisecond(re.ts);
                        if (curt.CompareTo(t1) < 0)
                        {
                            t1 = curt;
                        }
                        if (curt.CompareTo(t2) > 0)
                        {
                            t2 = curt;
                        }
                        if (curt.CompareTo(mint) < 0)
                        {
                            mint = curt;
                        }
                        if (curt.CompareTo(maxt) > 0)
                        {
                            maxt = curt;
                        }
                    }
                     

                    if (ss.facts.Count > 1)
                    {
                        facttime.Add(t2.Subtract(t1).TotalMinutes);
                    }
                    eventcnt += ss.facts.Count;
                }
            }

            Console.WriteLine("Facts cnt : {0}", factcnt);
            Console.WriteLine("Event cnt : {0}", eventcnt);
            Console.WriteLine("Facts avg : {0}", eventcnt*1.0/factcnt);
            Console.WriteLine("user cnt : {0}", user2factcnt.Count);
            Console.WriteLine("user max fact cnt : {0}", user2factcnt.Max(a=>a.Value));
            Console.WriteLine("Single Event max length : {0}", facttime.Max());
            Console.WriteLine("Single Event avg length : {0}", facttime.Average());
            Console.WriteLine("min time   : {0}", mint);
            Console.WriteLine("max time   : {0}", maxt);
        }

        public static void Stat02()
        {
            Dictionary<string, int> fid2usercnt = new Dictionary<string, int>();

            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\facts.json";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\fid2usercnt.csv";
            using (StreamReader rd = new StreamReader(infile))
            {
                int factcnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (factcnt++ % 10000 == 0)
                    {
                        Console.WriteLine(factcnt);
                    }

                    Facts ss = JsonConvert.DeserializeObject<Facts>(content);
 

                    HashSet<string> cfidset = new HashSet<string>();
                    foreach (var re in ss.facts)
                    {
                        if (cfidset.Contains(re.fid))
                        {
                            //.WriteLine("fid hit!!");
                        }
                        else
                        {
                            cfidset.Add(re.fid);
                        }
                         
                    }

                    foreach (var cfid in cfidset)
                    {
                        if (!fid2usercnt.ContainsKey(cfid))
                        {
                            fid2usercnt.Add(cfid, 1);
                        }
                        else
                        {
                            fid2usercnt[cfid]++;
                        }
                    } 
                }
            }

            Naive.OutputSortedDict(fid2usercnt, outfile);
        }

        public static void GenUriStat()
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\urls.csv";
            string[] outfiles = { 
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\url_freq_dep00.csv",
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\url_freq_dep01.csv",
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\url_freq_dep02.csv",
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\url_freq_dep03.csv"
                                 
                                 };
            Dictionary<string, int>[] url2freq00 = new Dictionary<string, int> [4];
            for (int i = 0; i < 4; i++)
            {
                url2freq00[i] = new Dictionary<string, int>();
            }


            using (StreamReader rd = new StreamReader(infile))
            {
                int cnt = 0; 
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 10000 == 0)
                    {
                        Console.WriteLine(cnt);
                    }
                    string[] words = content.Split(',');
                    string url = words[1];
                    if (url.IndexOf("?") > 0)
                    {
                        url = url.Substring(0, url.IndexOf("?"));
                    }
                    int dep = 0;
                    int slash_idx = url.IndexOf("/");
                    while (dep <= 3)
                    {
                        AddDict(url2freq00[dep], url.Substring(0, slash_idx < 0 ? url.Length : slash_idx));
                        dep++;
                        if (slash_idx < 0)
                            break;
                        slash_idx = url.IndexOf("/", slash_idx + 1);
                    }
                }
            }

            for (int i = 0; i < 4; i++)
            {
                Naive.OutputSortedDict(url2freq00[i], outfiles[i]);
            }
        }

        private static void AddDict(Dictionary<string, int> dict, string p)
        {
            if (!dict.ContainsKey(p))
            {
                dict.Add(p, 1);
            }
            else
            {
                dict[p]++;
            }
        }

       
        
        public static void TrainUserCount()
        {
            HashSet<string> userset = new HashSet<string>();
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");
            Console.WriteLine(user2matches.Count);
        }
    }
}
