using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIKM2016.Structure
{
    class Facts
    {
        public List<Record> facts;
        public string uid;
    }

    class Record
    {
        public string fid;
        public long ts;
    }
}
