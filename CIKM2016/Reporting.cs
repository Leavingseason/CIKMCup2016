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
    class Reporting
    {
        public static void Run(string[] args)
        {

           //   Naive.EnrichTrainingfile(int.Parse(args[0]));

            //Tools.FileMerger.MergeFiles(
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\merge",
            //     @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train_and_valid_feautre_ran_lr_tree",
            //false);

           // Naive.GetTrainfileCandida();

            //        Program.AppendUserFeatures(
            //@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train_tree_report_keyurls",
            // @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train_tree_report_keyurls_userfe"
            //);



            // Naive.GenTrainCandidateWithTreeModel(int.Parse(args[0]), int.Parse(args[1]));

            //PrepareTrainValidFiles();

            //MakeValidUid80NegSample(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr", @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr_80");
            //MakeValidUid80NegSample(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_tree", @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_tree_80");

            //CountLines(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\train_ranlr");
            //CountLines(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\train_ranlr");

            Test();

           // GetBasicStat();

            //double recall =  0.2268;
            //double precision = 0.6421; 
            //Console.WriteLine("F1 {0}", 2 * recall * precision / (recall + precision));

           // FilterModelRecall();

        }

        public static void FilterModelRecall()
        {
            string[] infiles = { 
                @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_tree\train_candi_report_LR_completelist0.csv" ,      
                @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_tree\train_candi_report_LR_completelist1.csv"         
                               };

            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_tree\LR_filter_evalution.csv";

            Dictionary<string, List<Tuple<int, float>>> user2candi = new Dictionary<string, List<Tuple<int, float>>>();
            foreach (var infile in infiles)
            {
                using (StreamReader rd = new StreamReader(infile))
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
                        if (!user2candi.ContainsKey(words[1]))
                        {
                            user2candi.Add(words[1], new List<Tuple<int,float>>());
                        }
                        user2candi[words[1]].Add(new Tuple<int, float>(int.Parse(words[0]), float.Parse(words[3])));
                    }
                }
            }

            foreach (var pair in user2candi)
            {
                pair.Value.Sort((a,b)=>b.Item2.CompareTo(a.Item2));
            }

            int usercnt = user2candi.Count;
            double explen = 0;
            int poscnt = 0 ;
            foreach (var pair in user2candi)
            {
                int len = pair.Value.Count;
                for(int i=0;i<len;i++){
                    if (pair.Value[i].Item1 == 1)
                    {
                        explen += i;
                        poscnt++;
                    }
                }
            }

            explen /= usercnt;

            Console.WriteLine("user cnt : {0}", usercnt);
            Console.WriteLine("avg pred len : {0}", explen);

            using (StreamWriter wt = new StreamWriter(outfile))
            {
                //int len = user2candi.Min(a => a.Value.Count);
                int hit = 0;
                for (int i = 0; i < 2000; i++)
                {
                    foreach (var pair in user2candi)
                    {
                        if (pair.Value[i].Item1 == 1)
                        {
                            hit++;
                        }
                    }
                    double recall = hit * 1.0 / poscnt;
                    wt.WriteLine("{0},{1}",i,recall);
                }
            }
        }


        public static void GetBasicStat(Dictionary<string, int> user2fidcnt, Dictionary<string, int> fid2usercnt)
        {
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\facts.json";


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


                    user2fidcnt.Add(ss.uid, ss.facts.Count);

                    ss.facts.Sort((a, b) => a.ts.CompareTo(b.ts));

                    foreach (var fid in ss.facts)
                    {
                        if (!fid2usercnt.ContainsKey(fid.fid))
                        {
                            fid2usercnt.Add(fid.fid, 1);
                        }
                        else
                        {
                            fid2usercnt[fid.fid]++;
                        }
                    }
                }
            }

        }


        private static void GetBasicStat()
        {
            Dictionary<string, int> user2fidcnt = new Dictionary<string, int>();
            Dictionary<string, int> fid2usercnt = new Dictionary<string, int>();
            GetBasicStat(user2fidcnt, fid2usercnt);

            var word2doc = Loader.LoadWord2Doccnt();
            var url002factcnt = Loader.LoadUrl2factcnt00();
            var url012factcnt = Loader.LoadUrl2factcnt01();
            var url022factcnt = Loader.LoadUrl2factcnt02();
            var url032factcnt = Loader.LoadUrl2factcnt03();

            Console.WriteLine("Total User Cnt: {0}", user2fidcnt.Count);
            Console.WriteLine("Total Fid Cnt: {0}", fid2usercnt.Count);
            Console.WriteLine("Max fidcnt per user: {0}", user2fidcnt.Max(a=>a.Value));
            Console.WriteLine("Min fidcnt per user: {0}", user2fidcnt.Min(a => a.Value));
            Console.WriteLine("Avg fidcnt per user: {0}", user2fidcnt.Average(a => a.Value));

            Console.WriteLine("Word Cnt : {0}", word2doc.Count);
            Console.WriteLine("URL dep0 cnt : {0}", url002factcnt.Count);
            Console.WriteLine("URL dep1 cnt : {0}", url012factcnt.Count);
            Console.WriteLine("URL dep2 cnt : {0}", url022factcnt.Count);
            Console.WriteLine("URL dep3 cnt : {0}", url032factcnt.Count);
        }

        private static void Test()
        {
            int topk = 1000;
            int s = 0;

            topk = 1500;
            SubmissionHelper.SelectTopInstances(topk, s, 2);
            SubmissionHelper.Evaluate(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\tlc\submission_" + topk + "_" + s + ".txt",
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr_80");


         
            topk = 1800;
            SubmissionHelper.SelectTopInstances(topk, s, 2);
            SubmissionHelper.Evaluate(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\tlc\submission_" + topk + "_" + s + ".txt",
 @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr_80");


             
            topk = 2100;
            SubmissionHelper.SelectTopInstances(topk, s, 30);
            SubmissionHelper.Evaluate(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\tlc\submission_" + topk + "_" + s + ".txt",
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr_80");

            topk = 2400;
            SubmissionHelper.SelectTopInstances(topk, s, 30);
            SubmissionHelper.Evaluate(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\tlc\submission_" + topk + "_" + s + ".txt",
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr_80");

            topk = 2700;
            SubmissionHelper.SelectTopInstances(topk, s, 30);
            SubmissionHelper.Evaluate(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\tlc\submission_" + topk + "_" + s + ".txt",
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr_80");

            topk = 3000;
            SubmissionHelper.SelectTopInstances(topk, s, 30);
            SubmissionHelper.Evaluate(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\tlc\submission_" + topk + "_" + s + ".txt",
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr_80");

            topk = 3300;
            SubmissionHelper.SelectTopInstances(topk, s, 30);
            SubmissionHelper.Evaluate(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\tlc\submission_" + topk + "_" + s + ".txt",
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr_80");

  
        }

        public static void MakeValidUid80NegSample(string infile, string outfile)
        {
            Dictionary<string,int> uid2negcnt = new Dictionary<string,int>();
            using(StreamReader rd = new StreamReader(infile))
            using(StreamWriter wt = new StreamWriter(outfile)){
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    string key = words[1] + "," + words[2];
                    if (words[0] == "1")
                    {
                        wt.WriteLine(content);
                    }
                    else
                    {
                        if (!uid2negcnt.ContainsKey(words[1]))
                        {
                            uid2negcnt.Add(words[1], 1);
                            wt.WriteLine(content);
                        }
                        else
                        {
                            if (uid2negcnt[words[1]] < 80)
                            {
                                wt.WriteLine(content);
                                uid2negcnt[words[1]]++;
                            }
                        }
                    }
                }
            }
        }

        public static void CountLines(string infile)
        {
            var cnt = 0;
            using (StreamReader rd = new StreamReader(infile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    cnt++;
                }
            }
            Console.WriteLine(cnt);
        }

        public static void PrepareTrainValidFiles()
        {
            HashSet<string> valid_uids = new HashSet<string>();
            using (StreamReader rd = new StreamReader(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_LR\for_valid\valid_rng1000"))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    valid_uids.Add(content);
                }
            }

            HashSet<string> LR_pairs = new HashSet<string>();
            DirectoryInfo lrDir = new DirectoryInfo(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_LR");
            foreach (var file in lrDir.GetFiles())
            {
                using (StreamReader rd = new StreamReader(file.FullName))
                {
                    string content = null;
                    while ((content = rd.ReadLine()) != null)
                    {
                        string[] words = content.Split(',');
                        string key = words[1] + "," + words[2];
                        if (!LR_pairs.Contains(key))
                        {
                            LR_pairs.Add(key);
                        }
                    }
                }
            }

            HashSet<string> tree_pairs = new HashSet<string>();
            DirectoryInfo treeDir = new DirectoryInfo(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_by_tree");
            var tree_candi_files = new List<string>();
            foreach (var file in treeDir.GetFiles())
            {
                if (file.Name.Length > "train_candi10.csv".Length)
                    continue;
                tree_candi_files.Add(file.FullName);
            }
            tree_candi_files.Add(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\candi\train_candi_tree0.csv");
            tree_candi_files.Add(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\candi\train_candi_tree1.csv");

            foreach (var file in tree_candi_files)
            {
                using (StreamReader rd = new StreamReader(file))
                {
                    string content = null;
                    while ((content = rd.ReadLine()) != null)
                    {
                        string[] words = content.Split(',');
                        string key = words[1] + "," + words[2];
                        if (!tree_pairs.Contains(key))
                        {
                            tree_pairs.Add(key);
                        }
                    }
                }
            }


            string outfile_traincomplete = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\train_complete_eq";
            string outfile_trainrandom = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\train_random";
            string outfile_trainran_lr = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\train_ranlr_eq";
            string outfile_valid_tree = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_tree"; 
            string outfile_valid_lr = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr";
            string outfile_valid_lr_tree = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\valid_lr_tree";

            string complete_instance_file = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train_and_valid_feautre_ran_lr_tree";

            HashSet<string> existing_pairs = new HashSet<string>();

            Random rng = new Random((int)DateTime.Now.Ticks);

            using(StreamReader rd = new StreamReader(complete_instance_file))
            using(StreamWriter wt01 = new StreamWriter(outfile_traincomplete))
            using(StreamWriter wt02 = new StreamWriter(outfile_trainrandom))
            using(StreamWriter wt03 = new StreamWriter(outfile_trainran_lr))
            using (StreamWriter wt04 = new StreamWriter(outfile_valid_tree))
            using(StreamWriter wt05 = new StreamWriter(outfile_valid_lr))
            using (StreamWriter wt06 = new StreamWriter(outfile_valid_lr_tree))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    string key = words[1] + "," + words[2];

                    if (existing_pairs.Contains(key))
                    {
                        continue;
                    }
                    else
                    {
                        existing_pairs.Add(key);
                    }

                    if (!valid_uids.Contains(words[1]))
                    {
                        if (words[0] == "1" || rng.NextDouble() < 0.34)//|| tree_pairs.Contains(key) || rng.NextDouble() < 0.26
                        {
                            wt01.WriteLine(content);
                        } 
                         

                        if ((!LR_pairs.Contains(key) && !tree_pairs.Contains(key) ) || words[0]=="1")
                        {
                            wt02.WriteLine(content);
                        }
                        else
                        {
                            if (rng.NextDouble() < 0.001)
                            {
                                wt02.WriteLine(content);
                            }
                        }

                        if ( words[0] == "1")
                        {
                            wt03.WriteLine(content);
                        } 
                        else
                        {
                            if (rng.NextDouble() < 0.001)
                            {
                                wt03.WriteLine(content);
                            }
                            else
                            {
                                if ((LR_pairs.Contains(key) || !tree_pairs.Contains(key)) && rng.NextDouble()<0.383)
                                {
                                    wt03.WriteLine(content);
                                }
                            }
                        }
                    }
                    else
                    {
                        wt06.WriteLine(content);
                        if (tree_pairs.Contains(key) || words[0] == "1")
                        {
                            wt04.WriteLine(content);
                        }
                        if (LR_pairs.Contains(key) || words[0] == "1")
                        {
                            wt05.WriteLine(content);
                        }
                    }
                }
            }
        }
    }
}
