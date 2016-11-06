using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIKM2016.Structure
{
    class TimeFeatureBag
    {
        public double[] hour2cnt;
        public double[] day2cnt;
        public double[] month2cnt;  // 2014-12   2016-07
        public Dictionary<string, int> date2cnt;
        public DateTime firstdate;
        public DateTime lastdate;
    }
}
