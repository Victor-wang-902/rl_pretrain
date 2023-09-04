import torch.multiprocessing as mp
import torch
def foo(q, e):
    a = torch.rand(10000000) / 1.0
    event.set()

results_queue = mp.Manager().Queue()
processes = []
event = mp.Event()

for worker_id in range(2):
    p = mp.Process(target=foo, args=(results_queue, event))
    #p = mp.Process(target=self.generate_worker, args=(worker_id, total_itr, max_length, batch_size // self.num_workers, results_queue))
    processes.append(p)
    p.start()
    #time.sleep(1)
#time.sleep(10)
#print("main starts to wait")
event.wait()
#generated_batches = []

#print(results_queue.empty())
#for _ in range(self.num_workers):
#while not results_queue.empty():
    #generated_batches.append(torch.tensor(results_queue.get()))
    #generated_batches.append(results_queue.get())
#print(generated_batches)
for p in processes:
    p.join()
    #p.terminate()

#generated = torch.cat(generated_batches, dim=0)
#return generated