using CIKM2016.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIKM2016
{
    class Program
    {
        static void Main(string[] args)
        {
            UserProfileInfer.Utils.Common.ShowFirstlines(
@"\\mlsdata\e$\Users\v-lianji\mlsdata\Dianping\gensim\doc2vec\document_line_filter100_0.2.txt",
@"\\mlsdata\e$\Users\v-lianji\mlsdata\Dianping\gensim\doc2vec\document_line_filter100_0.2_firstlines.txt",
1000);
            ////            UserProfileInfer.Utils.Common.ShowFirstlines(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\keyurl_enriched\tmp\test_part_keyurl10",
            ////@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\keyurl_enriched\tmp\test_part_keyurl10_firstlines.csv", 10000);

            // EarlyJobs.GenTestFileWithLinearModel();
            // EarlyJobs.GenTestFile();

            //EarlyJobs.GenUriStat();

            //Console.WriteLine(UserProfileInfer.Utils.Common.ParseTimeStampMillisecond(1419870366865673));

            // EarlyJobs.StatUserFidCnt(); 


            //  Naive.EnrichTrainingfile(int.Parse(args[0]));
            //Naive.GenTrainCandidateWithTreeModel(int.Parse(args[0]), int.Parse(args[1]));
            // Naive.GenTimeFeatures();
            // Naive.GetValidfileCandida();

            //  SmallJobs();

            //            Tools.FileMerger.MergeFiles(
            //@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\TLC\FT4k_0.05_keyurls_exttraintree",
            //@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\pred_merge_tlc\FT4k_0.05_keyurls_exttraintree.tsv", true);


            ///   SubmissionHelper.SelectTopInstances(105000, 2);
            //double pre = 0;
            //double rec = 0.449;
            //Console.WriteLine(2.0 * pre * rec / (pre + rec));

            //  SplitDataSet.SplitTrainValidFile();
            // 

            //  UrlEnrich();

            //  SplitDataSet.ExtendTestCandi();

            //EarlyJobs.ValidDetection();

            // Naive.ExtractUserFeatures();

            //        for (int i = 0; i < 9; i++)
            //        {
            //            AppendUserFeatures(
            //@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\new_added_from_tree\train_tree_part" + i,
            //@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\new_added_from_tree\train_tree_userfe_part" + i
            //);
            //        }
            //        AppendUserFeatures(
            //@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\new_added_from_tree\merge\train_newadded_tree_keyurls",
            // @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\new_added_from_tree\merge\train_newadded_tree_keyurls_userfe"
            //);

            //EnrichWithModelPred(
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\TLC\LR\LR_train_urls_sym_part1.inst.txt",
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\TLC\FR\train_urls_sym_part1.inst.txt",
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\complete_keyurls_split\train_urls_sym_part1",
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\complete_keyurls_split\train_urls_sym_part1_TLCPred"
            //    );

            // EnrichWithModelPred(
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\TLC\LR\LR_test_all_urls.inst.txt",
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\TLC\FR\FR_test_all_urls.inst.txt",
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\Group4Ranking\test_all_urls",
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\Group4Ranking\test_all_urls_TLCPred"
            //    );

            //SplitDataSet.SelectInstanceByValidSet(
            //    new string[] { @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part1_keyurl.csv",
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part2_keyurl.csv",
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part4_keyurl.csv", 
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\train_0_3_5_6_7_8_9_10_11_keyurl.csv"
            //    },
            //   @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\holdout_valid\train_valid_keyurl_complete"
            //    );

            //SplitDataSet.SelectInstanceByValidSet(
            //    new string[] { @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\train_part_1_2_4_keyurl_userfe.csv" },
            //   @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\holdout_valid\train_part_1_2_4_keyurl_userfe.csv"
            //    );
            //SplitDataSet.SelectInstanceByValidSet(
            //    new string[] { @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\train_part_1_2_4_keyurl_userfe.csv" },
            //   @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\holdout_valid\train_part_1_2_4_keyurl_userfe_removed.csv",
            //   true
            //    );

            //            Ensemble.Merge(new string[]{
            //            @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\holdout_valid\TLC\FT.inst.txt",
            //            @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\holdout_valid\TLC_userfe\FT.inst.txt",
            //            },
            //            new double[] { 1.0, 10 },
            //            @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\holdout_valid\TLC\ensemble.tsv"
            //            );

          




            //SplitDataSet.RemoveDuuplicateLines(
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\holdout_valid\train_valid_keyurl_complete",
            //    @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\holdout_valid\train_valid_keyurl_complete_nodup"
            //    );
            //SplitDataSet.RemoveDuuplicateLines(
            //   @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\holdout_valid\train_valid_keyurl_complete_userfe",
            //   @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\holdout_valid\train_valid_keyurl_complete_userfe_nodup"
            //   );

            //  SplitDataSet.ReduceTrainfile();

            // ExpPipeline();

          //  Reporting.Run(args);

            Console.WriteLine("Mission complete.");
            Console.ReadKey();
        }

        private static void ExpPipeline()
        {
            SplitDataSet.SelectKInstance(3000);
        }

        public static void AppendUserFeatures(string infile, string outfile)
        {
            Dictionary<string, float[]> user2featureline = new Dictionary<string, float[]>();
            using (StreamReader rd = new StreamReader(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\user_features.csv"))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split(',');
                    float[] features = new float[words.Length - 1];
                    for (int i = 0; i < features.Length; i++)
                    {
                        features[i] = float.Parse(words[i+1]);
                    }
                    user2featureline.Add(words[0], features);
                }
            }

            using (StreamReader rd = new StreamReader(infile))
            using(StreamWriter wt =new StreamWriter(outfile))
            {
                string content = null;
                int cnt = 0;
                int errcnt = 0;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 100000 == 0)
                    {
                        Console.WriteLine(cnt);
                    }
                    string[] words = content.Split(',');
                    if (!user2featureline.ContainsKey(words[1]) || !user2featureline.ContainsKey(words[2]))
                    {
                        errcnt++;
                        continue;
                    }
                    var fa = user2featureline[words[1]];
                    var fb = user2featureline[words[2]];
                    wt.Write(content);
                    int len = fa.Length;
                    for (int i = 0; i < len; i++)
                    {
                        wt.Write(",{0}", Math.Min(fa[i], fb[i]));
                    }
                    wt.WriteLine();
                }
            }

        }

        private static void UrlEnrich()
        {
            var user2url = UrlComAnalysis.LoadUser2url(1);
//            UrlComAnalysis.AppendKeyUrlFeature(
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\keyurl_enriched\test_part12"  ,
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\keyurl_enriched\tmp\test_part_keyurl12"  ,
//user2url
//);

//            UrlComAnalysis.AppendKeyUrlFeature(
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part4",
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\train_part4_keyurl.csv",
//user2url
//);

//            UrlComAnalysis.AppendKeyUrlFeature(
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\train_0_3_5_6_7_8_9_10_11.csv",
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\train_0_3_5_6_7_8_9_10_11_keyurl.csv", user2url
//);
            for (int i = 12; i < 18; i++)
            {
                UrlComAnalysis.AppendKeyUrlFeature(
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\keyurl_enriched\test_part" + i,
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\keyurl_enriched\tmp\test_part_keyurl" + i,
user2url
);
            }
        }



        public static void EnrichWithModelPred(string LRPredfile, string FRPredfile, string infile, string outfile)
        {
            Dictionary<string, double> key2pred_lr = LoadTLCOutput(LRPredfile);
            Dictionary<string, double> key2pred_fr = LoadTLCOutput(FRPredfile);

            using (StreamReader rd = new StreamReader(infile))
            using(StreamWriter wt= new StreamWriter(outfile))
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
                    string key = words[1] + "|" + words[2];
                    if (!key2pred_fr.ContainsKey(key) || !key2pred_lr.ContainsKey(key))
                    {
                        throw new Exception(); 
                    }
                    wt.WriteLine("{0},{1},{2}", content, key2pred_lr[key], key2pred_fr[key]);
                }
            }
        }

        private static Dictionary<string, double> LoadTLCOutput(string LRPredfile)
        {
            Dictionary<string, double> res = new Dictionary<string, double>();
            using (StreamReader rd = new StreamReader(LRPredfile))
            {
                string content = rd.ReadLine();
                while ((content = rd.ReadLine()) != null)
                {
                    string[] words = content.Split('\t');
                    if(!res.ContainsKey(words[0]))
                        res.Add(words[0], double.Parse(words[2]));
                }
            }
            return res;
        }

        public static void SmallJobs()
        {
           // EarlyJobs.UrlComAnalysis_random();
           // UrlComAnalysis.GenUrlLiftRatio();
            //Tools.Evaluation.CalcLiftCurve(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\TLC02\FT10k.inst.txt", @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\TLC02\FT10k.inst.eva.csv");
           //SplitDataSet.SplitTrainingSet(500);
            Tools.FileMerger.MergeFiles(
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\tmp",
@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\train_0_3_5_6_7_8_9_10_11_keyurl_exttraintree_userfe",
false);
            //CandiMerger.MergeAndLabelling(new string[] { @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features\train_enriched_labeled.csv", @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_hq_candi.csv" }
            //    , @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\train_all_candi.csv"
            //    );

//           GroupSplitFiles(
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\Group4Ranking\test_all_urls",
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\Group4Ranking\split",
//9
//);
           // Tools.FileSpliter.SplitFiles(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\extend_train_random50\train_ranpair.csv", @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\candi\extend_train_random50\split", 9, false);

          //  Naive.GetTrainfileCandida();


          //  SplitDataSet.IncludeAllHQTrainCandi();


//            AppendSym2TrainFile(
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\complete_keyurls_split\train_urls_part0",
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\complete_keyurls_split\train_urls_sym_part0"
//);
//            AppendSym2TrainFile(
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\complete_keyurls_split\train_urls_part1",
//@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\feature_split\merge\complete_keyurls_split\train_urls_sym_part1"
//);

        }

        public static void AppendSym2TrainFile(string infile, string outfile)
        {
            Dictionary<string, string> key2line = new Dictionary<string, string>();

            using (StreamWriter wt = new StreamWriter(outfile))
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
                        string key = words[1] + "," + words[2];
                        if (words[0]=="1" && !key2line.ContainsKey(key))
                        {
                            key2line.Add(key, content);
                        }
                        wt.WriteLine(content);
                    }
                }

                foreach (var pair in key2line)
                {
                    string[] tokens = pair.Key.Split(',');
                    string key = tokens[1] + "," + tokens[0];
                    if (!key2line.ContainsKey(key))
                    {
                        //wt.WriteLine(pair.Value);
                        string[] words = pair.Value.Split(',');
                        words[1] = tokens[1];
                        words[2] = tokens[0];
                        wt.WriteLine(string.Join(",",words));
                    }
                }
            }
        }

        private static void GroupSplitFiles(string infile, string outpath, int k)
        {
            Random rng = new Random((int)DateTime.Now.Ticks);
            Dictionary<string, int> uid2idx = new Dictionary<string, int>();
            StreamWriter[] wts = new StreamWriter[k];
            
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
                    if (!uid2idx.ContainsKey(words[1]))
                    {
                        int idx = rng.Next(k);
                        uid2idx.Add(words[1], idx);
                    }
                }
            }

            using (StreamReader rd = new StreamReader(infile))
            {
                for (int i = 0; i < k; i++)
                {
                    wts[i] = new StreamWriter(outpath + "\\part" + i); ;
                }

                int cnt = 0;
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 100000 == 0)
                    {
                        Console.WriteLine(cnt);
                    }
                    string[] words = content.Split(',');
                    wts[uid2idx[words[1]]].WriteLine(content);
                }

                for (int i = 0; i < k; i++)
                {
                    wts[i].Close();
                }
            }
        }
    }
}
