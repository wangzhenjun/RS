# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
import math
import copy
import matplotlib.pyplot as plt

def initData(sigma, mu1, mu2, k, n):
	global X
	global mu
	global expections
	X = zeros((1, n))
	mu = random.random(k)
	expections = zeros((n, k))
	for i in range(n):
		if random.random(1) > 0.5:
			X[0, i] = random.normal(mu1, sigma)
		else:
			X[0, i] = random.normal(mu2, sigma)
	print("************")
	print("初始化观测数据X:")
	print(X)

def eStep(sigma, k, n):
	global expections
	global mu
	global X
	for i in range(n):
		Denom = 0
		Numer = [0.0] * k
		for j in range(k):
			Numer[j] = math.exp((-1/(2*(float(sigma**2))))*(float(X[0, i]- mu[j]))**2)
			Denom += Numer[j]
		for j in range(k):
			expections[i, j] = Numer[j] / Denom
	print('************')
	print('隐藏变量E(Z)')
	print(expections)

def mStep(k, n):
	global expections
	global X
	for j in range(k):
		Numer = 0
		Denom = 0
		for i in range(n):
			Numer += expections[i, j]*X[0, i]
			Denom += expections[i, j]
		mu[j] = Numer / Denom

def run(sigma, mu1, mu2, k, n, iter_num, epsilon):
	initData(sigma, mu1, mu2, k, n)
	for i in range(iter_num):
		old_mu = copy.deepcopy(mu)
		eStep(sigma, k, n)
		mStep(k, n)
		print(i, mu)
		if sum(abs(mu-old_mu)) < epsilon:
			break

if __name__ == '__main__':
	sigma = 5
	mu1 = 10
	mu2 = 26
	k = 2
	n = 1000
	iter_num = 1000
	epsilon = 0.00001
	run(sigma, mu1, mu2, k, n, iter_num, epsilon)
