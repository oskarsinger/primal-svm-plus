import numpy as np

class HyperBandOptimizer:

    def __init__(self,
        get_sample,
        get_evaluation,
        max_iter=81,
        eta=3):

        self.get_sample = get_sample
        self.get_evaluation = get_evaluation
        self.max_iter = max_iter
        self.eta = eta

        self.s_max = int(
            np.log(self.max_iter) / np.log(self.eta))
        self.B = (self.s_max + 1) * self.max_iter
        self.theta = None

    def get_parameters(self):

        return self.theta

    def run(self):

        samples = None
        best = []

        for s in reversed(range(self.s_max+1)):

            print('HyperBand Outer Iteration:', self.s_max - s)

            n = int(np.ceil(self.eta**s * self.B / self.max_iter / (s + 1)))
            r = np.ceil(self.max_iter * self.eta**(-s))

            print('\tNum Samples:', n, 'Max Iters:', r)

            samples = [self.get_sample() for _ in range(n)] 

            for i in range(s + 1):

                print('\tHyperBand Inner Iteration:', i)

                n_i = np.floor(n * self.eta**(-i))
                r_i = r * self.eta**i

                evals = [self.get_evaluation(sample, r_i)
                         for sample in samples]
                argsorted = np.argsort(evals)
                num_to_keep = int(n_i / self.eta)

                if num_to_keep > 0:
                    samples = [samples[j] for j in argsorted[-num_to_keep:]]
                else:
                    best_sample = samples[argsorted[0]]
                    best_evaluation = evals[argsorted[0]]

                    print('Sample:', best_sample)
                    print('Evaluation:', best_evaluation)

                    best.append((best_sample, best_evaluation))

        best = [(sample, self.get_evaluation(sample, self.max_iter))
                for (sample, _) in best]
        sorted_by_eval = sorted(best, key=lambda x:x[1])
        (self.theta, self.theta_eval) = sorted_by_eval[-1]

        print('Final Sample:', self.theta)
        print('Final Eval:', self.theta_eval)
