Run:
make
./simpush -f dblp -qn 50

Evaluation:
g++ -std=c++11 cal_evalaute.cpp -o eval
bash run_dblp_eval.sh

original paper
@article{DBLP:journals/pvldb/ShiJYXY20,
  author    = {Jieming Shi and
               Tianyuan Jin and
               Renchi Yang and
               Xiaokui Xiao and
               Yin Yang},
  title     = {Realtime Index-Free Single Source SimRank Processing on Web-Scale
               Graphs},
  journal   = {PVLDB},
  volume    = {13},
  number    = {7},
  pages     = {966--978},
  year      = {2020}
}
