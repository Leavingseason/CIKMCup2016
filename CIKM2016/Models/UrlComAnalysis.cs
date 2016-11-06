using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIKM2016.Models
{
    class UrlComAnalysis
    {
        public static void AppendKeyUrlFeature(string infile, string outfile , Dictionary<string, HashSet<string>> user2urls)
        {
            if (user2urls == null)
            {
                user2urls = LoadUser2url(1);
            }
            
            var keyurls = LoadKeyUrls();
            int dep = keyurls.Count;

            string default_urlfeature = "";
            for (int i = 0; i < dep; i++)
            {
                default_urlfeature += ",0";
            }

            Console.WriteLine("Data preparation completed.");
            Console.WriteLine("url dep: {0}", dep);

            using(StreamReader rd = new StreamReader(infile))
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                int lcnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {  
                    if (lcnt++ % 100000 == 0)
                    {
                        Console.WriteLine(lcnt);
                    }
                    string[] words = content.Split(',');
                    string subcontent = string.Join(",", words, 0, words.Length - 6);
                    if (user2urls.ContainsKey(words[1]) && user2urls.ContainsKey(words[2]))
                    {
                        wt.Write(subcontent);
                        for (int i = 0; i < dep; i++)
                        {
                            int cnt = 0;
                            foreach (var keyurl in keyurls[i].Keys)
                            {
                                if (user2urls[words[1]].Contains(keyurl) && user2urls[words[2]].Contains(keyurl))
                                {
                                    cnt++;
                                }
                            }
                            wt.Write(","+cnt);
                        }
                        wt.WriteLine();
                    }
                    else
                    {
                        wt.WriteLine(subcontent + default_urlfeature);
                    }
                }
            }
        }

        public static Dictionary<string, HashSet<string>> LoadUser2url(int target_dep, Dictionary<string, Structure.Facts> user2fact = null)
        {  
            if( user2fact ==null) 
                user2fact = Loader.LoadUserFacts();

            var fact2url = Loader.LoadFid2Url(target_dep);

            Dictionary<string, HashSet<string>> user2urls = new Dictionary<string, HashSet<string>>();

            
            HashSet<string> uid_set = new HashSet<string>(user2fact.Keys);


            foreach (var uid in uid_set)
            {
                user2urls.Add(uid, new HashSet<string>());
                foreach (var fact in user2fact[uid].facts)
                {
                    if (!fact2url.ContainsKey(fact.fid))
                    {
                        continue;
                    }

                    string target_url = fact2url[fact.fid];
                     
                    if (!user2urls[uid].Contains(target_url))
                    {
                        user2urls[uid].Add(target_url);
                    }
                }
            }

            return user2urls;
        }

        public static List<Dictionary<string,double>> LoadKeyUrls()
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\common_user_stat\url_lift.csv";
            Dictionary<string, double> url2comlift = UserProfileInfer.Utils.Common.LoadDict<string, double>(infile, ',');
            List<Dictionary<string, double>> res = new List<Dictionary<string, double>>();
            double[] levels = { 678260, 84782, 22634, 8813, 4281, 2139 };
            for (int i = 0; i < levels.Length; i++)
            {
                Dictionary<string, double> filter_urls = new Dictionary<string, double>();
                foreach (var pair in url2comlift)
                {
                    if (pair.Value > levels[i])
                    {
                        filter_urls.Add(pair.Key, pair.Value);
                    }
                }
                res.Add(filter_urls);
            }
            return res;
        }

        public static void GenUrlLiftRatio()
        {
            double matchBase = 506000;
            double ranBase = 330000.0 * 20000;

            string matchfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\common_user_stat\url02.csv";
            string ranfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\common_user_stat\url_random_2w.csv";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\common_user_stat\url_lift.csv";

            Dictionary<string, double> url2ratio_match = LoadUrl2Cnt(matchfile, matchBase, 20);
            Dictionary<string, double> url2ratio_ran = LoadUrl2Cnt(ranfile, ranBase, 0);

            Dictionary<string, double> url2lift = new Dictionary<string, double>();
            foreach (var pair in url2ratio_match)
            {
                if (url2ratio_ran.ContainsKey(pair.Key))
                {
                    url2lift.Add(pair.Key, pair.Value / url2ratio_ran[pair.Key]);
                }
                else
                {
                    url2lift.Add(pair.Key, 19.9);
                }
            }

            Naive.OutputSortedDict(url2lift, outfile);
        }

        private static Dictionary<string, double> LoadUrl2Cnt(string matchfile, double matchBase, int threshold)
        {
            Dictionary<string, double> t = UserProfileInfer.Utils.Common.LoadDict<string, double>(matchfile, ',', false, 0, 1);
            Dictionary<string, double> res = new Dictionary<string, double>();
            foreach (var pair in t)
            {
                if (pair.Value >= threshold)
                {
                    res.Add(pair.Key, pair.Value / matchBase);
                }
            }
            return res;
        }
    }
}
