from BaesyanNetwork import BayesianNet
import numpy as np
from MaximumLikelihood import MLE, arrayMLE
from KLDivergenceCalculation import calculateKLDivergency, meanKL


#Calculate KL divergency for more datasets of lenght 10 learning parameter from ech trhough Maximum Likelihood
array10 = np.array([[BayesianNet, '10Cases.csv'], [BayesianNet, '10_1Cases.csv'], [BayesianNet, '10_2Cases.csv'],
                        [BayesianNet, '10_3Cases.csv'], [BayesianNet, '10_4Cases.csv'], [BayesianNet, '10_5Cases.csv'],
                        [BayesianNet, '10_6Cases.csv'], [BayesianNet, '10_7Cases.csv'], [BayesianNet, '10_8Cases.csv'],
                        [BayesianNet, '10_9Cases.csv'], [BayesianNet, '10_10Cases.csv']])

arrayMLE(array10)

#Calculate KL divergency for more datasets of lenght 50 learning parameter from ech trhough Maximum Likelihood
array50 = np.array([[BayesianNet, '50Cases.csv'], [BayesianNet, '50_1Cases.csv'], [BayesianNet, '50_2Cases.csv'],
                        [BayesianNet, '50_3Cases.csv'], [BayesianNet, '50_4Cases.csv'], [BayesianNet, '50_5Cases.csv'],
                        [BayesianNet, '50_6Cases.csv'], [BayesianNet, '50_7Cases.csv'], [BayesianNet, '50_8Cases.csv'],
                        [BayesianNet, '50_9Cases.csv'], [BayesianNet, '50_10Cases.csv']])

arrayMLE(array50)

#Calculate KL divergency for more datasets of lenght 100 learning parameter from ech trhough Maximum Likelihood
array100 = np.array([[BayesianNet, '100Cases.csv'], [BayesianNet, '100_1Cases.csv'], [BayesianNet, '100_2Cases.csv'],
                        [BayesianNet, '100_3Cases.csv'], [BayesianNet, '100_4Cases.csv'], [BayesianNet, '100_5Cases.csv'],
                        [BayesianNet, '100_6Cases.csv'], [BayesianNet, '100_7Cases.csv'], [BayesianNet, '100_8Cases.csv'],
                        [BayesianNet, '100_9Cases.csv'], [BayesianNet, '100_10Cases.csv']])

arrayMLE(array100)

#Calculate MLE for all other datasets
bNet250 = BayesianNet()
csvfile250 = '250Cases.csv'
MLE(csvfile250, bNet250)

bNet500 = BayesianNet()
csvfile500 = '500Cases.csv'
MLE(csvfile500, bNet500)

bNet750 = BayesianNet()
csvfile750 = '750Cases.csv'
MLE(csvfile750, bNet750)

bNet1000 = BayesianNet()
csvfile1000 = '1000Cases.csv'
MLE(csvfile1000, bNet1000)

bNet1250 = BayesianNet()
csvfile1250 = '1250Cases.csv'
MLE(csvfile1250, bNet1250)

bNet1500 = BayesianNet()
csvfile1500 = '1500Cases.csv'
MLE(csvfile1500, bNet1500)

bNet1750 = BayesianNet()
csvfile1750 = '1750Cases.csv'
MLE(csvfile1750, bNet1750)

bNet2000 = BayesianNet()
csvfile2000 = '2000Cases.csv'
MLE(csvfile2000, bNet2000)

bNet2250 = BayesianNet()
csvfile2250 = '2250Cases.csv'
MLE(csvfile2250, bNet2250)

bNet2500 = BayesianNet()
csvfile2500 = '2500Cases.csv'
MLE(csvfile2500, bNet2500)

bNet3000 = BayesianNet()
csvfile3000 = '3000Cases.csv'
MLE(csvfile3000, bNet3000)

bNet3500 = BayesianNet()
csvfile3500 = '3500Cases.csv'
MLE(csvfile3500, bNet3500)

bNet4000 = BayesianNet()
csvfile4000 = '4000Cases.csv'
MLE(csvfile4000, bNet4000)

bNet4500 = BayesianNet()
csvfile4500 = '4500Cases.csv'
MLE(csvfile4500, bNet4500)

bNet5000 = BayesianNet()
csvfile5000 = '5000Cases.csv'
MLE(csvfile5000, bNet5000)

bNet6000 = BayesianNet()
csvfile6000 = '6000Cases.csv'
MLE(csvfile6000, bNet6000)

bNet7000 = BayesianNet()
csvfile7000 = '7000Cases.csv'
MLE(csvfile7000, bNet7000)

bNet8000 = BayesianNet()
csvfile8000 = '8000Cases.csv'
MLE(csvfile8000, bNet8000)

bNet9000 = BayesianNet()
csvfile9000 = '9000Cases.csv'
MLE(csvfile9000, bNet9000)

bNet10000 = BayesianNet()
csvfile10000 = '10000Cases.csv'
MLE(csvfile10000, bNet10000)


#print All Kullback-Leibler for all learned network
print 'KL divergence for n = 10',meanKL(array10)
print 'KL divergence for n = 50', meanKL(array50)
print 'KL divergence for n = 100',meanKL(array100)
print 'KL divergence for n = 250', calculateKLDivergency(bNet250)
print 'KL divergence for n = 500', calculateKLDivergency(bNet500)
print 'KL divergence for n = 750', calculateKLDivergency(bNet750)
print 'KL divergence for n = 1000', calculateKLDivergency(bNet1000)
print 'KL divergence for n = 1250', calculateKLDivergency(bNet1250)
print 'KL divergence for n = 1500', calculateKLDivergency(bNet1500)
print 'KL divergence for n = 1750', calculateKLDivergency(bNet1750)
print 'KL divergence for n = 2000', calculateKLDivergency(bNet2000)
print 'KL divergence for n = 2250', calculateKLDivergency(bNet2250)
print 'KL divergence for n = 2500', calculateKLDivergency(bNet2500)
print 'KL divergence for n = 3000', calculateKLDivergency(bNet3000)
print 'KL divergence for n = 3500', calculateKLDivergency(bNet3500)
print 'KL divergence for n = 4000', calculateKLDivergency(bNet4000)
print 'KL divergence for n = 4500', calculateKLDivergency(bNet4500)
print 'KL divergence for n = 5000', calculateKLDivergency(bNet5000)
print 'KL divergence for n = 6000', calculateKLDivergency(bNet6000)
print 'KL divergence for n = 7000', calculateKLDivergency(bNet7000)
print 'KL divergence for n = 8000', calculateKLDivergency(bNet8000)
print 'KL divergence for n = 9000', calculateKLDivergency(bNet9000)
print 'KL divergence for n = 10000', calculateKLDivergency(bNet10000)



