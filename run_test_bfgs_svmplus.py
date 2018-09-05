import click
import os

import numpy as np

from prinfo.testers import Li2016BFGSTester as L2016BFGST
from prinfo.loaders import GaussianLoader as GL
from prinfo.loaders import LinearSVMPlusLoader as LSVMPL
from prinfo.loaders import ARDSSubsampledEHRLUPILoader as ARDSSEHRLUPIL
from prinfo.servers import BatchServer as BS

@click.command()
@click.option('--n', default=1000)
@click.option('--do', default=100)
@click.option('--dp', default=80)
@click.option('--c', default=100.0)
@click.option('--gamma', default=10)
@click.option('--max-rounds', default=200)
@click.option('--epsilon', default=10**(-5))
@click.option('--max-distance', default=5)
@click.option('--ards-dir', default=None)
def run_things_all_day_bb(
    n,
    do,
    dp,
    c,
    gamma,
    max_rounds,
    epsilon,
    max_distance,
    ards_dir):

    server = None
    loader = None

    if ards_dir is None:
        o_loader = GL(n, do)
        p_loader = GL(n, dp)
        loader = LSVMPL(
            o_loader,
            p_loader,
            max_distance_from_margin=max_distance)
        server = BS(loader)
    else:
        ards_path = os.path.join(
            ards_dir, 'bin_train.csv')
        loader = ARDSSEHRLUPIL(ards_path)
        server = BS(loader)

        server.get_data()

        n = server.rows()

    tester = L2016BFGST(
        server, 
        c, 
        gamma,
        max_rounds=max_rounds,
        epsilon=epsilon)

    tester.run()

    (X_o, X_p, y) = server.get_data()
    alphas = tester.get_parameters()[:n]
    w_hat = np.dot((alphas * y).T, X_o).T
    y_hat = np.sign(np.dot(X_o, w_hat))
    evaluation = get_bce(y, y_hat)

    print(tester.objectives[::100])
    print('train')
    print('\n'.join(
        [k + ':' + str(v) 
         for (k, v) in evaluation.items()]))

    test_loader = None

    if ards_dir is not None:
        test_ards_path = os.path.join(
            ards_dir, 
            'bin_test.csv')
        test_loader = ARDSSEHRLUPIL(test_ards_path)
    else:
        test_o_loader = GL(n, do)
        test_p_loader = GL(n, dp)
        w_o = loader.w_o
        w_p = loader.w_p
        test_loader = LSVMPL(
            test_o_loader,
            test_p_loader,
            max_distance_from_margin=max_distance,
            w_o_init=w_o,
            w_p_init=w_p)

    test_server = BS(test_loader)
    (test_X_o, test_X_p, test_y) = test_server.get_data()
    test_y_hat = np.sign(np.dot(test_X_o, w_hat))
    test_evaluation = get_bce(test_y, test_y_hat)

    print('test')
    print('\n'.join(
        [k + ':' + str(v) 
         for (k, v) in test_evaluation.items()]))

if __name__=='__main__':
    run_things_all_day_bb()
