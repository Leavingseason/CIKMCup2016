using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIKM2016.Models
{
    class SubmissionHelper
    {
        public static void SelectTopInstances_V0(int k, int s)
        {
            Dictionary<string, List<Tuple<string, double>>> user2matches = new Dictionary<string,List<Tuple<string,double>>>();
            List<Tuple<string, string, double>> globals = new List<Tuple<string, string, double>>();
            HashSet<string> visited = new HashSet<string>();
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features\test_enriched_labeled_LR.csv";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\pred\submission_" + k + "_" + s + ".txt";

            int cnt = 0;
            using (StreamReader rd = new StreamReader(infile))
            {
                string content = null;
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 100000 == 0)
                    {
                        Console.WriteLine(cnt);
                    }
                    string[] words = content.Split(',');
                    double score = double.Parse(words[2]);
                    globals.Add(new Tuple<string, string, double>(words[0], words[1], score));
                    if (!user2matches.ContainsKey(words[0]))
                    {
                        user2matches.Add(words[0], new List<Tuple<string, double>>());
                    }
                    user2matches[words[0]].Add(new Tuple<string, double>(words[1], score));
                }
            }

            globals.Sort((a, b) => b.Item3.CompareTo(a.Item3));

            using (StreamWriter wt = new StreamWriter(outfile))
            {
                foreach (var tuple in globals)
                {
                    string key = tuple.Item1 + "," + tuple.Item2;
                    if (!visited.Contains(key))
                    {
                        visited.Add(key);
                        wt.WriteLine(key);
                    }
                    k--;
                    if (k <= 0)
                    {
                        break;
                    }
                }

                if (s > 0)
                {
                    foreach (var pair in user2matches)
                    {
                        var matches = pair.Value;
                        matches.Sort((a, b) => b.Item2.CompareTo(a.Item2));
                        for (int i = 0; i < s && i < matches.Count; i++)
                        {
                            string key = pair.Key + "," + matches[i].Item1;
                            if (!visited.Contains(key))
                            {
                                visited.Add(key);
                                wt.WriteLine(key);
                            }
                        }
                    }
                }
            }
        }

        public static void Evaluate(string predfile, string gtfile)
        {
            Dictionary<string, int> gt = new Dictionary<string, int>();

            using (StreamReader rd = new StreamReader(gtfile))
            {
                string content = rd.ReadLine();
                while ((content = rd.ReadLine()) != null)
                { 
                    string[] words = content.Split(','); 
                    string key = words[1] + "," + words[2];
                    int label = int.Parse(words[0]);
                    if (!gt.ContainsKey(key) && label==1)
                    {
                        gt.Add(key, label);
                    }
                }
            }

            int hit = 0;
            int cnt = 0;
            using (StreamReader rd = new StreamReader(predfile))
            {
                string content = rd.ReadLine();
                while ((content = rd.ReadLine()) != null)
                {
                    cnt++;
                    if (gt.ContainsKey(content))
                    {
                        hit++;
                    }
                }
            }
            double recall = hit*1.0/gt.Count;
            double precision = hit * 1.0 / cnt;
            Console.WriteLine("Recall {0}", recall);
            Console.WriteLine("Precision {0}",precision );
            Console.WriteLine("F1 {0}", 2 * recall * precision / (recall + precision));
        }

        public static void SelectTopInstances(int k, int s, double longen_ratio = 2)
        {
            Console.WriteLine(k + "\t" + s);
            Dictionary<string, List<Tuple<string, double>>> user2matches = new Dictionary<string, List<Tuple<string, double>>>();
            List<Tuple<string, string, double>> globals = new List<Tuple<string, string, double>>();
            HashSet<string> visited = new HashSet<string>();
           // string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\features03\test_feature_split\pred_merge_tlc\FT15k_0.05_keyurls.tsv";
            //string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\pred\V2\submission_" + k + "_" + s + ".txt";
            string infile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\tlc\63.inst.txt";
            string outfile = @"\\mlsdata\e$\Users\v-lianji\others\CIKM16\my\Data4Report\features\train-valid\tlc\submission_" + k + "_" + s + ".txt";

            int cnt = 0;
            using (StreamReader rd = new StreamReader(infile))
            {
                string content = rd.ReadLine();
                while ((content = rd.ReadLine()) != null)
                {
                    if (cnt++ % 100000 == 0)
                    {
                      //  Console.WriteLine(cnt);
                    }
                    string[] words = content.Split('\t');
                    double score = double.Parse(words[3]);
                    string[] tokens = words[0].Split('|');
                    if (tokens.Length > 2)
                    {
                        throw new Exception();
                    }
                    globals.Add(new Tuple<string, string, double>(tokens[0], tokens[1], score));

                    if (!user2matches.ContainsKey(tokens[0]))
                    {
                        user2matches.Add(tokens[0], new List<Tuple<string, double>>());
                    }
                    user2matches[tokens[0]].Add(new Tuple<string, double>(tokens[1], score));

                    if (!user2matches.ContainsKey(tokens[1]))
                    {
                        user2matches.Add(tokens[1], new List<Tuple<string, double>>());
                    }
                    user2matches[tokens[1]].Add(new Tuple<string, double>(tokens[0], score));
                }
            }

            globals.Sort((a, b) => b.Item3.CompareTo(a.Item3));
            

            int longen_idx = (int)( k * longen_ratio);
            if (longen_idx > globals.Count)
            {
                longen_idx = globals.Count - 1; // k + 100000;
            }
            double longen_score = globals[longen_idx].Item3;

            using (StreamWriter wt = new StreamWriter(outfile))
            {
                foreach (var tuple in globals)
                {
                    string key = tuple.Item1 + "," + tuple.Item2;
                    if (!visited.Contains(key))
                    {
                        visited.Add(key);
                        wt.WriteLine(key);
                    }
                    k--;
                    if (k <= 0)
                    {
                        break;
                    }
                }

                if (s > 0)
                {
                    foreach (var pair in user2matches)
                    {
                        var matches = pair.Value;
                        matches.Sort((a, b) => b.Item2.CompareTo(a.Item2));
                        for (int i = 0; i < s && i < matches.Count; i++)
                        {
                            string key = pair.Key + "," + matches[i].Item1;
                            if (!visited.Contains(key) && matches[i].Item1.CompareTo(pair.Key) > 0 && matches[i].Item2 > longen_score)
                            {
                                visited.Add(key);
                                wt.WriteLine(key);
                            }
                        }
                    }
                }
            }
        }
    }
}
