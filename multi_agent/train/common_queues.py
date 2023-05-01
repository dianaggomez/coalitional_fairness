import numpy as np

for i in ["0", "02", "04" , "06", "08", "1"]:
  path = "/home/diana/journal/multi_agent/train/models/alpha_{}/successful_queues_{}.npy".format(i,i)
  queue = np.load(path)
  print(len(queue))
  if i == "0":
    print(i)
    new_queues = queue
    # print(new_queues)
  else:
    final_queues = []
    for q in queue:
      # print(new_queues)
      for nw_q in new_queues:
        if (nw_q == q).all():
          final_queues.append(q)

    new_queues = final_queues

print("No. of queue in common varying alpha: ", len(final_queues))

np.save("final_queues.npy", final_queues)

# print(len(np.load("/home/diana/journal/multi_agent/train/final_queues.npy")))