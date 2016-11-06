using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIKM2016
{
    class CandiMerger
    {
        public static void MergeAndLabelling(string[] infiles, string outfile)
        {
            HashSet<string> visited = new HashSet<string>();
            var user2matches = Loader.LoadGroundTruth(@"\\mlsdata\e$\Users\v-lianji\others\CIKM16\data-train-dca\train.csv");
            using (StreamWriter wt = new StreamWriter(outfile))
            {
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
                            if (words[1].CompareTo(words[2]) >= 0)
                            {
                                continue;
                            }
                            string key = words[1] + "," + words[2];
                            if (visited.Contains(key))
                            {
                                continue;
                            }
                            int label = 0;
                            if (user2matches.ContainsKey(words[1]) && user2matches[words[1]].Contains(words[2]))
                            {
                                label = 1;
                            }
                            wt.WriteLine("{0},{1},{2}", label, words[1], words[2]);
                        }
                    }
                }

                foreach (var pair in user2matches)
                {
                    string uid = pair.Key;
                    foreach (var muid in pair.Value)
                    {
                        if (uid.CompareTo(muid) >= 0)
                        {
                            continue;
                        }
                        string key = uid + "," + muid;
                        if (visited.Contains(key))
                        {
                            continue;
                        }
                        wt.WriteLine("{0},{1},{2}", 1, uid, muid);
                    }
                }
            }

           
        }
    }
}
