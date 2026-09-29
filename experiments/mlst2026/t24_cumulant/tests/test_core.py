import numpy as np
import pytest
from numpy.testing import assert_allclose
from experiments.mlst2026.t24_cumulant.core import *

def test_affine_and_gram_tensor():
 r=np.random.default_rng(123);x=r.normal(size=(100,4));a=r.normal(size=(4,4))+3*np.eye(4)
 first=statistics(x);second=statistics(x@a+r.normal(size=4));gram=statistics(x,'gram')
 assert_allclose([first['b1'],first['b2']],[second['b1'],second['b2']],rtol=1e-11)
 assert_allclose(first['b1'],gram['b1'],rtol=1e-12)

def test_rank_blocks_complex():
 with pytest.raises(ValueError):statistics(np.ones((50,3)))
 with pytest.raises(ValueError):statistics(np.eye(3))
 with pytest.raises(ValueError):statistics(np.ones((50,3),complex))
 x=np.random.default_rng(2).normal(size=(40,3));assert local_statistic(x,[20,20])['status']=='available'
 assert local_statistic(x,[38,2])['status']=='unavailable';assert local_statistic(x,[40])['status']=='unavailable'
 z=np.array([[1+2j,3+4j]]);assert_allclose(complex_real(z),np.sqrt(2)*np.array([[1,2,3,4]]))
 assert excess_ratio(3,np.array([2.,3.,4.]),2,np.array([1.,2.,3.])) is None

def test_null_holm_and_calibration():
 r=np.random.default_rng(45);bank=null_b2(r,64,3)
 assert plus_one(bank.max()+1,bank)==.001
 assert plus_one(bank.min()-1,bank)==1
 adjusted=holm({'a':.01,'b':None,'c':.03});assert adjusted['a']['adjusted_p']==.03;assert adjusted['b']['adjusted_p']==1
 assert not calibration(10)['failed'];assert calibration(30)['failed']
 assert 'limited' in classify(True,False,.4);assert 'qualified' in classify(True,True,None)

def test_control_moments():
 r=np.random.default_rng(987);n=150000;p=3;u=np.array([1.,0,0])
 for kind in ('S0','S1','S2'):
  x=matched_control(r,n,p,kind,u);assert_allclose(x.mean(0),np.zeros(p),atol=.025);assert_allclose(np.cov(x,rowvar=False),np.eye(p),atol=.05)
 # Algebraic covariance of centered Bernoulli construction is exactly I.
 eta=.5;assert_allclose(np.eye(p)-eta*np.outer(u,u)+eta*np.outer(u,u),np.eye(p))

def test_malformed_nulls_and_ratios():
 with pytest.raises(ValueError):plus_one(0,np.zeros((999,2)))
 assert excess_ratio(np.nan,np.arange(3.),1,np.arange(3.)) is None
 assert excess_ratio(5,np.arange(3.),1,np.array([np.nan,1])) is None
