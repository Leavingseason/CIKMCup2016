using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIKM2016.Models
{
    class SplitDataSet
    {
        public static void ReduceTrainfile()
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\product\train_0_3_5_6_7_8_9_10_11_keyurl_exttraintree_userfe";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\product\train_0_3_5_6_7_8_9_10_11_keyurl_exttraintree_userfe_small02";
            Random rng = new Random();
            using (StreamReader rd = new StreamReader(infile))
            using(StreamWriter wt =new StreamWriter(outfile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    if (words[0] == "1" || rng.NextDouble() < 0.7)
                    {
                        wt.WriteLine(content);
                    }
                }
            }
        }

        public static void RemoveDuuplicateLines(string infile, string outfile)
        {
            int cnt=0;
            int hit=0;
            HashSet<string> visited = new HashSet<string>();
            using (StreamReader rd = new StreamReader(infile))
            using(StreamWriter wt = new StreamWriter(outfile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    cnt++;
                    string[] words = content.Split(',');
                    string key = words[1] + "," + words[2];
                    if (visited.Contains(key))
                    {
                        hit++;
                    }
                    else
                    {
                        visited.Add(key);
                        wt.WriteLine(content);
                    }
                }
            }
            Console.WriteLine("{0} / {1}", hit, cnt);
        }

        public static void SelectInstanceByValidSet(string[] infiles, string outfile, bool remove=false)
        {
            HashSet<string> valid_set = new HashSet<string>();
            using (StreamReader rd = new StreamReader(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_LR\for_valid\valid_rng1000"))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    valid_set.Add(content);
                }
            }

            using (StreamWriter wt = new StreamWriter(outfile))
            {
                foreach (var file in infiles)
                {
                    using (StreamReader rd = new StreamReader(file))
                    {
                        int cnt = 0;
                        string content = null;
                        while ((content = rd.ReadLine()) != null)
                        {
                            if (cnt++ % 100000 == 0)
                            {
                                Console.WriteLine(cnt);
                            }
                            string[] words = content.Split(',');
                            if (remove)
                            {
                                if (!valid_set.Contains(words[1]) && !valid_set.Contains(words[2]))
                                {
                                    wt.WriteLine(content);
                                }
                            }
                            else
                            {
                                if (valid_set.Contains(words[1]))
                                {
                                    wt.WriteLine(content);
                                }
                            }
                        }
                    }
                }
            }
        }

        public static void SelectKInstance(int k)
        {
            List<string> uids = new List<string>();
            using (StreamReader rd = new StreamReader(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_LR\for_valid\valid_lr_positive"))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    uids.Add(content);
                }
            }

            HashSet<string> visited = new HashSet<string>();
            using (StreamReader rd = new StreamReader(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_LR\for_valid\valid_rng" + 1000))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    visited.Add(content);
                }
            }


            Random rng = new Random();
            using (StreamWriter wt = new StreamWriter(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_LR\for_valid\valid_rng" + k))
            {
                int cnt = 0;
                while (cnt < k)
                {
                    int idx = rng.Next(uids.Count);
                    if (visited.Contains(uids[idx]))
                    {
                        continue;
                    }
                    wt.WriteLine(uids[idx]);
                    uids.RemoveAt(idx);
                    cnt++;
                }
            }
        }

        public static void SelectValidInstance()
        {
           // Dictionary<string, int> train_lr_candi = new Dictionary<string, int>();
            HashSet<string> train_lr_candi = new HashSet<string>();
            string path = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_LR";
            DirectoryInfo dir = new DirectoryInfo(path);
            foreach (var file in dir.GetFiles())
            {
                using (StreamReader rd = new StreamReader(file.FullName))
                {
                    string content = null;
                    while ((content = rd.ReadLine()) != null)
                    {
                        string[] words = content.Split(',');
                        if (words[0] == "1")
                        {
                            if (!train_lr_candi.Contains(words[1]))
                            {
                                train_lr_candi.Add(words[1]);
                            }
                        }
                    }
                }
            }


            HashSet<string> train_part1_candi = new HashSet<string>();
            string part1_file = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\complete_keyurls_split\train_urls_part1";
            using (StreamReader rd = new StreamReader(part1_file))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    if (words[0] == "1")
                    {
                        if (!train_part1_candi.Contains(words[1]))
                        {
                            train_part1_candi.Add(words[1]);
                        }
                    }
                }
            }

            HashSet<string> train_userfe_candi = new HashSet<string>();
            string userfe_file = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\train_part_1_2_4_keyurl_userfe.csv";
            using (StreamReader rd = new StreamReader(userfe_file))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    if (words[0] == "1")
                    {
                        if (!train_userfe_candi.Contains(words[1]))
                        {
                            train_userfe_candi.Add(words[1]);
                        }
                    }
                }
            }

            List<string> candi = new List<string>();
            foreach (var uid in train_userfe_candi)
            {
                if (train_lr_candi.Contains(uid) && train_part1_candi.Contains(uid))
                {
                    candi.Add(uid);
                }
            }

            using (StreamWriter wt = new StreamWriter(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_LR\for_valid\valid_lr_positive"))
            {
                foreach (var uid in candi)
                {
                    wt.WriteLine(uid);
                }
            }
        }

        public static void IncludeAllHQTrainCandi()
        {
            HashSet<string> hqpairs = new HashSet<string>();
            string hq_train_candi_file = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_hq_candi.csv";
            using (StreamReader rd = new StreamReader(hq_train_candi_file))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    string key = words[1] + "," + words[2];
                    if (!hqpairs.Contains(key))
                    {
                        hqpairs.Add(key);
                    }
                }
            }

            //List<string> hqlines = new List<string>();
            //string[] train_lines = {
            //                       @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part1_keyurl.csv",
            //                       @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part2_keyurl.csv",
            //                       @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part4_keyurl.csv",
            //                       @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part5_keyurl.csv",
            //                       @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part7_keyurl.csv",
            //                       @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part8_keyurl.csv",
            //                      //
            //                       };

            //foreach (var file in train_lines)
            //{
            //    using (StreamReader rd = new StreamReader(file))
            //    {
            //        int cnt = 0;
            //        string content = null;
            //        while ((content = rd.ReadLine()) != null)
            //        {
            //            if (cnt++ % 100000 == 0)
            //            {
            //                Console.WriteLine(cnt);
            //            }
            //            string[] words = content.Split(',');
            //            string key = words[1] + "," + words[2];
            //            if (hqpairs.Contains(key))
            //            {
            //                hqlines.Add(content);
            //            }
            //        }
            //    }
            //}

            //Console.WriteLine("hq line count : {0}", hqlines.Count);
            Random rng = new Random((int)DateTime.Now.Ticks);
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\train_allhq_ran0.3_keyurl_hqtrain.csv";
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                using (StreamReader rd = new StreamReader(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\train_0_3_5_6_7_8_9_10_11_keyurl_hqtrain.csv"))
                {
                    int cnt = 0;
                    string content = null;
                    while ((content = rd.ReadLine()) != null)
                    {
                        if (cnt++ % 100000 == 0)
                        {
                            Console.WriteLine(cnt);
                        }
                        string[] words = content.Split(',');
                        string key = words[1] + "," + words[2];
                        if (hqpairs.Contains(key) || rng.NextDouble() < 0.3)
                        {
                            wt.WriteLine(content);
                        }
                    }
                }
                //foreach (var line in hqlines)
                //{
                //    wt.WriteLine(line);
                //}
            }
        }

        public static void ExtendTestCandi()
        {
            string old_infile = @"D:\tmp\test_part\test_instance_largen\pred_FT400.tsv";
            string wait_folder = @"D:\tmp\test_part\test_instance_largen\new_waiting_list";
            string outfile = @"D:\tmp\test_part\test_instance_largen\update_complete_test_inst_2delete.csv";

            HashSet<string> existing_pairs = new HashSet<string>();
            int cnt = 0;
            using (StreamReader rd = new StreamReader(old_infile))
            {
                string content = rd.ReadLine();
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 100000 == 0)
                    {
                        Console.WriteLine(cnt);
                    }
                    string[] words = content.Split('\t');

                    string[] tokens = words[0].Split('|');
                    string pairkey = tokens[0] + "," + tokens[1];

                    existing_pairs.Add(pairkey);
                }
            }

            cnt = 0;
            int hit = 0;
            DirectoryInfo waitdir = new DirectoryInfo(wait_folder);
            using (StreamWriter wt = new StreamWriter(outfile))
            {
                foreach (var file in waitdir.GetFiles())
                {
                    using (StreamReader rd = new StreamReader(file.FullName))
                    {
                        string content = null;
                        while ((content = rd.ReadLine()) != null)
                        {
                            if (cnt++ % 100000 == 0)
                            {
                                Console.WriteLine("{0}  / {1}", hit, cnt);
                            }
                            string[] words = content.Split(',');
                            string pairkey = words[0] + "," + words[1];
                            if (!existing_pairs.Contains(pairkey))
                            {
                               // existing_pairs.Add(pairkey);
                                wt.WriteLine(pairkey);
                            }
                            else
                            {
                                hit++;
                            }
                        }
                    }
                }
            }

             

        }

        public static void SplitTrainValidFile()
        {
            string validfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\valid.csv";
            string trainallfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features02\train_all_02.csv";

            string outtrainfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features02\train-valid\mytrain.csv";
            string outvalidfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features02\train-valid\myvalid.csv";

            HashSet<string> valid_uids = new HashSet<string>();
            using (StreamReader rd = new StreamReader(validfile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    if (!valid_uids.Contains(words[0]))
                    {
                        valid_uids.Add(words[0]);
                    }
                    if (!valid_uids.Contains(words[1]))
                    {
                        valid_uids.Add(words[1]);
                    }
                }
            }

            using (StreamReader rd = new StreamReader(trainallfile))
            using(StreamWriter wt00 = new StreamWriter(outtrainfile))
            using (StreamWriter wt01 = new StreamWriter(outvalidfile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    string a = words[1];
                    string b = words[2];
                    if (valid_uids.Contains(a) || valid_uids.Contains(b))
                    {
                        wt01.WriteLine(content);
                    }
                    else
                    {
                        wt00.WriteLine(content);
                    }
                }
            }
        }

        public static void SplitTrainingSet(int k)
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv";
            string outfile00 = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\valid.csv";
            string outfile01 = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\train.csv";

            Dictionary<string, List<string>> user2matches = new Dictionary<string, List<string>>();

            using (StreamReader rd = new StreamReader(infile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    AddLink(words[0], words[1], user2matches);
                    AddLink(words[1], words[0], user2matches);
                }
            }

            Random rng = new Random((int)DateTime.Now.Ticks);
            List<string> uids = user2matches.Keys.ToList();
            HashSet<string> res = new HashSet<string>();
            int cover = 0;
            while (cover < k)
            {
                int idx = rng.Next(uids.Count);
                if (res.Contains(uids[idx]))
                {
                    continue;
                }
                List<string> queue = new List<string>();
                queue.Add(uids[idx]);
                 
                while (queue.Count > 0)
                {
                    string curuid = queue[0];
                    queue.RemoveAt(0);
                    if (res.Contains(curuid))
                    {
                        continue;
                    }

                    cover++;
                    res.Add(curuid);
                   
                    foreach (var cmatch in user2matches[curuid])
                    {
                        if (!res.Contains(cmatch))
                        {
                            queue.Add(cmatch);
                        }
                    }
                }
            }

            Console.WriteLine("valid user cnt: {0}\t{1}", res.Count,cover);

            using (StreamWriter wt = new StreamWriter(outfile00))
            {
                foreach (var uid in res)
                {
                    foreach (var match in user2matches[uid])
                    {
                        if (uid.CompareTo(match) < 0)
                        {
                            wt.WriteLine("{0},{1}",uid,match);
                        }
                    }
                }
            }
            using (StreamWriter wt = new StreamWriter(outfile01))
            {
                foreach (var uid in uids)
                {
                    if (res.Contains(uid))
                        continue;
                    foreach (var match in user2matches[uid])
                    {
                        if (uid.CompareTo(match) < 0 && !res.Contains(match))
                        {
                            wt.WriteLine("{0},{1}", uid, match);
                        }
                    }
                }
            }
        }

        private static void AddLink(string a, string b, Dictionary<string, List<string>> user2matches)
        {
            if (!user2matches.ContainsKey(a))
            {
                user2matches.Add(a, new List<string>());
            }
            user2matches[a].Add(b);
        }
    }
}
