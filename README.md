# Grey-box-attack
## 单样本优化中第二大步第一小步
考虑ori-adv-sim(embedding 层面和 cross-entropy层面) and fluency去finetune模型
## 存在问题
1. 对于相似性和流畅性的考虑是要在样本生成后才行的，而样本生成中用到的argmax会导致梯度无法传回adv-sample-generate-model。（gumble-softmax?）\\
2. 对于相似性，目前考虑gumble-softmax后得到词向量平均再计算，这个应该能解决。对于流畅性的梯度反向传播还有些问题。
