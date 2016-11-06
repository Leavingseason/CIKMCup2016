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
    class Loader
    {
        public static Dictionary<string, int> LoadFid2Usercnt()
        {
            return UserProfileInfer.Utils.Common.LoadDict<string, int>(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\fid2usercnt.csv", ',', false, 0, 1);
        }

        public static Dictionary<string, int> LoadWord2Doccnt()
        {
            return UserProfileInfer.Utils.Common.LoadDict<string, int>(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\word_freq_title.csv", ',', false, 0, 1);
        }

        public static Dictionary<string, int> LoadUrl2factcnt00()
        {
            return UserProfileInfer.Utils.Common.LoadDict<string, int>(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\url_freq_dep00.csv", ',', false, 0, 1);
        }
        public static Dictionary<string, int> LoadUrl2factcnt01()
        {
            return UserProfileInfer.Utils.Common.LoadDict<string, int>(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\url_freq_dep01.csv", ',', false, 0, 1);
        }
        public static Dictionary<string, int> LoadUrl2factcnt02()
        {
            return UserProfileInfer.Utils.Common.LoadDict<string, int>(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\url_freq_dep02.csv", ',', false, 0, 1);
        }
        public static Dictionary<string, int> LoadUrl2factcnt03()
        {
            return UserProfileInfer.Utils.Common.LoadDict<string, int>(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\url_freq_dep03.csv", ',', false, 0, 1);
        }

        public static Dictionary<string, Facts> LoadUserFacts(Dictionary<string, HashSet<string>> user2matches = null)
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\facts.json";

            Dictionary<string, Facts> user2fact = new Dictionary<string, Facts>();

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

                    if (user2matches != null && !user2matches.ContainsKey(ss.uid))
                    {
                        continue;
                    }

                    user2fact.Add(ss.uid, ss);

                    ss.facts.Sort((a, b) => a.ts.CompareTo(b.ts));
                }
            }

            return user2fact;
        }

        
        public static Dictionary<string, List<string>> LoadFid2Words()
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\titles.csv";
            Dictionary<string, List<string>> res = new Dictionary<string, List<string>>();
            int cnt = 0;
            using (StreamReader rd = new StreamReader(infile))
            {
                
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 100000 == 0)
                    {
                        Console.WriteLine(cnt + "\tLoadFid2Words");
                    }
                    string[] words = content.Split(',');
                    //HashSet<string> tokens = new HashSet<string>(words[1].Split(' '));
                    if (!res.ContainsKey(words[0]))
                    {
                        res.Add(words[0], new List<string>(words[1].Split(' ')));
                    }
                }
            }
            return res;
        }

        public static Dictionary<string, List<string>> LoadFid2Url()
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\urls.csv";
            Dictionary<string, List<string>> res = new Dictionary<string, List<string>>();

            using (StreamReader rd = new StreamReader(infile))
            {
                int cnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 10000 == 0)
                    {
                        Console.WriteLine(cnt + "\tLoadFid2Url");
                    }
                    string[] words = content.Split(',');
                    List<string> cur_urls = new List<string>(); 
                    string url = words[1];
                    if (url.IndexOf("?") > 0)
                    {
                        url = url.Substring(0, url.IndexOf("?"));
                    }
                    int dep = 0;
                    int slash_idx = url.IndexOf("/");
                    while (dep <= 3)
                    {
                        cur_urls.Add(url.Substring(0, slash_idx < 0 ? url.Length : slash_idx));
                        dep++;
                        if (slash_idx < 0)
                            break;
                        slash_idx = url.IndexOf("/", slash_idx + 1);
                    }

                    if (!res.ContainsKey(words[0]))
                    {
                        res.Add(words[0], cur_urls);
                    }
                }
            }

            return res;
        }

        public static Dictionary<string, string> LoadFid2Url(int d)
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\urls.csv";
            Dictionary<string, string> res = new Dictionary<string, string>();

            using (StreamReader rd = new StreamReader(infile))
            {
                int cnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 10000 == 0)
                    {
                        Console.WriteLine(cnt + "\tLoadFid2Url");
                    }
                    string[] words = content.Split(',');
                    List<string> cur_urls = new List<string>();
                    string url = words[1];
                    if (url.IndexOf("?") > 0)
                    {
                        url = url.Substring(0, url.IndexOf("?"));
                    }
                    int dep = 0;
                    int slash_idx = url.IndexOf("/");
                    string curl = ""; 
                    while (dep <= d)
                    {
                        if(dep==d)
                            curl = url.Substring(0, slash_idx < 0 ? url.Length : slash_idx);
                        dep++;
                        if (slash_idx < 0)
                        {
                            break;
                        }
                        slash_idx = url.IndexOf("/", slash_idx + 1);
                    }
                    if (!string.IsNullOrEmpty(curl))
                    {
                        res.Add(words[0], curl);
                    }

                }
            }

            return res;
        }

        public static Dictionary<string, HashSet<string>> LoadGroundTruth(string infile)
        {
            Dictionary<string, HashSet<string>> user2matches = new Dictionary<string, HashSet<string>>();
            using (StreamReader rd = new StreamReader(infile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    if (!user2matches.ContainsKey(words[0]))
                    {
                        user2matches.Add(words[0], new HashSet<string>());
                    }
                    if (!user2matches[words[0]].Contains(words[1]))
                    {
                        user2matches[words[0]].Add(words[1]);
                    }

                    if (!user2matches.ContainsKey(words[1]))
                    {
                        user2matches.Add(words[1], new HashSet<string>());
                    }
                    if (!user2matches[words[1]].Contains(words[0]))
                    {
                        user2matches[words[1]].Add(words[0]);
                    }
                }
            }
            return user2matches;
        }
    }
}
