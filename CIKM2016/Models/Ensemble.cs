using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIKM2016.Models
{
    class Ensemble
    {
        public static void Merge(string[] infiles, double[] weighted, string outfile)
        {
            Dictionary<string, double> key2values = new Dictionary<string, double>();
            for(int idx = 0; idx<infiles.Length;idx++)
            {
                string infile = infiles[idx];
                using (StreamReader rd = new StreamReader(infile))
                {
                    List<Tuple<string, double>> curlist = new List<Tuple<string, double>>();
                    string content = rd.ReadLine();
                    while ((content = rd.ReadLine()) != null)
                    {  
                        string[] words = content.Split('\t');
                        double score = double.Parse(words[2]);
                        curlist.Add(new Tuple<string, double>(words[0], score));
                    }
                    curlist.Sort((a, b) => b.Item2.CompareTo(a.Item2));
                    int cnt = curlist.Count;
                    for (int i = 0; i < cnt; i++)
                    {
                        double v = (cnt-i) * weighted[idx] / cnt;
                        if (!key2values.ContainsKey(curlist[i].Item1))
                        {
                            if (idx > 0)
                            {
                                throw new Exception();
                            }
                            key2values.Add(curlist[i].Item1, v);
                        }
                        else
                        {
                            key2values[curlist[i].Item1] += v;
                        }
                    }
                }
                
            }

            List<Tuple<string, double>> res  = new List<Tuple<string, double>>();
            foreach (var pair in key2values)
            {
                res.Add(new Tuple<string, double>(pair.Key, pair.Value));
            }
           // res.Sort((a, b) => a.Item2.CompareTo(b.Item2));

            using (StreamWriter wt = new StreamWriter(outfile))
            {
                wt.WriteLine("header");
                for (int i = 0; i < key2values.Count; i++)
                {
                    wt.WriteLine("{0}\t{1}\t{2}", res[i].Item1, 0, res[i].Item2);
                }
            }
        }
    }
}
