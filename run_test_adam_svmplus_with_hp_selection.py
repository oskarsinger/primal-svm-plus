import click
import os

import numpy as np

from prinfo.testers import Li2016SDCATesterWithHPSelection as L2016SDCAT
from prinfo.testers import Li2016AdamTesterWithHPSelection as L2016AdamT
from prinfo.loaders import LinearSVMPlusLoader as LSVMPL
from prinfo.loaders import GaussianLoader as GL
from prinfo.loaders import BernoulliLoader as BL
from prinfo.loaders import LinearDynamicsSequenceLoader as LDSL
from prinfo.loaders import ARDSSubsampledEHRLUPILoader as ARDSSEHRLUPIL
from prinfo.loaders import ARDSSubsampledEHRLoader as ARDSSEHRL
from prinfo.servers import BatchServer as BS
from prinfo.testers.utils import get_binary_classification_eval as get_bce
from prinfo.utils import get_rotation
from prinfo.optimizers import ParallelHyperBandOptimizer as PHBO

@click.command()
@click.option('--n', default=1000)
@click.option('--do', default=100)
@click.option('--dp', default=80)
@click.option('--p', default=0.7)
@click.option('--num-batches', default=None)
@click.option('--fold-k', default=5)
@click.option('--max-rounds', default=10000)
@click.option('--epsilon', default=10**(-5))
@click.option('--max-distance', default=100)
@click.option('--ards-dir', default=None)
@click.option('--num-processes', default=None)
@click.option('--mixture', default=False)
def run_things_all_day_bb(
    n,
    do,
    dp,
    p,
    num_batches,
    fold_k,
    max_rounds,
    epsilon,
    max_distance,
    ards_dir,
    num_processes,
    mixture):

    server = None
    loader = None

    if ards_dir is None:
        o_loader = GL(n, do)
        p_loader = BL(n, dp, p=p)
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

    get_zero_order = None

    if num_processes is not None:
        get_zero_order = lambda get_hps, get_vl: PHBO(
            get_hps, get_vl,
            max_iter=int(max_rounds/100),
            num_processes=int(num_processes))

    tester = None

    if num_batches is None:
        tester = L2016AdamT(
            server,
            fold_k=fold_k,
            max_rounds=max_rounds,
            epsilon=epsilon,
            get_zero_order=get_zero_order,
            mixture=mixture)
    else:
        tester = L2016SDCAT(
            server, 
            num_batches=int(num_batches),
            fold_k=fold_k,
            max_rounds=max_rounds,
            epsilon=epsilon,
            get_zero_order=get_zero_order,
            mixture=mixture)

    tester.run()

    # Get predictions on training set
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
        pre_P = np.random.randn(2*do, do) 
        (_, P) = np.linalg.eig(np.dot(pre_P.T, pre_P))
        R = get_rotation(0.09 * np.pi, P, P_inv=P.T)
        test_o_loader = LDSL(R, n, seed=5*np.ones((do,1)))
        test_p_loader = BL(n, dp, p=p)
        w_o = loader.w_o
        w_p = loader.w_p
        test_loader = LSVMPL(
            test_o_loader,
            test_p_loader,
            max_distance_from_margin=max_distance,
            w_o_init=w_o,
            w_p_init=w_p)

    test_server = BS(test_loader)
    (test_X, _, test_y) = test_server.get_data()
    test_y_hat = np.sign(np.dot(test_X, w_hat))
    test_evaluation = get_bce(test_y, test_y_hat)

    print('test')
    print('\n'.join(
        [k + ':' + str(v) 
         for (k, v) in test_evaluation.items()]))

if __name__=='__main__':
    run_things_all_day_bb()
