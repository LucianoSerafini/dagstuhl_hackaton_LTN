import numpy as np
import tensorflow as tf
import logictensornetworks as ltn

from pylab import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ltn.default_layers = 4
ltn.default_smooth_factor = 1e-10
ltn.default_tnorm = "product"
ltn.default_aggregator = "min"
ltn.default_optimizer = "rmsprop"
ltn.default_clauses_aggregator = "min"
ltn.default_positive_fact_penality = 0

number_of_features = 3

person = ltn.Domain(number_of_features, label="Person")

couple = ltn.Domain(number_of_features*2,label="Couple")

A = ltn.Predicate("A",person)
B = ltn.Predicate("B",person)
R = ltn.Predicate("R",couple)

a1_features = [1.0,0.0,0.0]
a2_features = [0.7,0.0,0.2]
a3_features = [0.9,0.2,0.1]

b1_features = [0.0,1.0,1.0]
b2_features = [0.1,0.8,0.7]
b3_features = [0.3,1.0,0.8]


a1 = ltn.Constant("a1",domain=person,value=a1_features)
a2 = ltn.Constant("a2",domain=person,value=a2_features)
a3 = ltn.Constant("a2",domain=person,value=a3_features)

b1 = ltn.Constant("b1",domain=person,value=b1_features)
b2 = ltn.Constant("b2",domain=person,value=b2_features)
b3 = ltn.Constant("b3",domain=person,value=b3_features)

c = ltn.Constant("c", domain=person)
d = ltn.Constant("d", domain=person)

c_features_are_in_01 = ltn.Clause([ltn.Literal(True,
                                               ltn.In_range(person, -0.4+np.zeros(number_of_features),
                                                                0.4+np.ones(number_of_features),sharpness=5.0), c)], label="c_is_in_0_1")
d_features_are_in_01 = ltn.Clause([ltn.Literal(True,
                                               ltn.In_range(person, -0.4+np.zeros(number_of_features),
                                                                0.4+np.ones(number_of_features),sharpness=5.0), d)], label="d_is_in_0_1")

A_a1 = ltn.Clause([ltn.Literal(True,A,a1)],label="A_a1")
A_a2 = ltn.Clause([ltn.Literal(True,A,a2)],label="A_a2")
A_a3 = ltn.Clause([ltn.Literal(True,A,a3)],label="A_a3")

B_b1 = ltn.Clause([ltn.Literal(True,B,b1)],label="B_b1")
B_b2 = ltn.Clause([ltn.Literal(True,B,b2)],label="B_b2")
B_b3 = ltn.Clause([ltn.Literal(True,B,b3)],label="B_b3")

A_or_B_c = ltn.Clause([ltn.Literal(True, A, c),
                       ltn.Literal(True, B, c)], label="A_c1_v_B_c1")

A_imp_not_B = ltn.Clause([ltn.Literal(False,A,person),
                          ltn.Literal(False,B,person)],label="A_imp_not_B")

couple_and_a1d = ltn.Domain_union([couple, ltn.Domain_concat([a1, d])])

Rxy_imp_Ax  = ltn.Clause([ltn.Literal(False, R, couple_and_a1d),
                          ltn.Literal(True, A, ltn.Domain_slice(couple_and_a1d, 0, 3))],
                         label="Rxy_imp_Ax")

Rxy_imp_By  = ltn.Clause([ltn.Literal(False, R, couple_and_a1d),
                          ltn.Literal(True, B, ltn.Domain_slice(couple_and_a1d, 3, 6))],
                         label="Rxy_imp_By")

couples_in_R = ltn.Domain(number_of_features*2,label="Couples_in_R")

Ra1d = ltn.Clause([ltn.Literal(True,R,ltn.Domain_concat([a1,d]))],label="R_of_a1_and_d")

positive_examples_on_R = ltn.Clause([ltn.Literal(True,R,couples_in_R)],
                                    label="Positive_examples_of_R")


KB = ltn.KnowledgeBase("KebAB", [A_a1, A_a2, A_a3,
                                 B_b1, B_b2, B_b3,
                                 A_or_B_c,
                                 A_imp_not_B,
                                 c_features_are_in_01,
                                 d_features_are_in_01,
                                 Rxy_imp_Ax,
                                 Rxy_imp_By,
                                 Ra1d,
                                 positive_examples_on_R
                                 ], ".")

data = np.array([[i,j,k] for i in np.linspace(0,1,20,endpoint=True)
                for j in np.linspace(0,1,20,endpoint=True)
                for k in np.linspace(0,1,20,endpoint=True)],dtype=np.float32)

data_pairs = np.array([np.concatenate([data[i],data[j]])
                                for i in np.random.choice(range(len(data)),70)
                                for j in np.random.choice(range(len(data)),70)])

feed_dict = {person.tensor:data,
             couple.tensor:data_pairs,
             couples_in_R.tensor:data_pairs[np.where(np.all(
                 data_pairs[:,:1] + 0.5 <= data_pairs[:,3:]))]}

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sat_level = sess.run(KB.tensor,feed_dict=feed_dict)
print "initialization", sat_level

while sat_level < 1e-10:
    sess.run(init)
    sat_level = sess.run(KB.tensor, feed_dict=feed_dict)
    print "initialization",sat_level
print(0, " ------> ", sat_level)

for i in range(5000):
  KB.train(sess,feed_dict=feed_dict)
  sat_level = sess.run(KB.tensor,feed_dict=feed_dict)
  print(i + 1, " ------> ", sat_level)
  if sat_level > .99:
      break
KB.save(sess)

# Classifying new individual

print "1,0,1 is of type A with degree =",sess.run(ltn.Literal(True,A,person).tensor,{person.tensor:[[1.,0.,1.]]})
print "0,1,1 is of type A with degree =",sess.run(ltn.Literal(True,A,person).tensor,{person.tensor:[[0.,1.,1.]]})

# Generating new features for c1...c3

c_features = sess.run(c.tensor)[0]
print "the features of c are", c_features
print "c is of type A with degree =", sess.run(ltn.Literal(True, A, c).tensor)
print "c is of type B with degree =", sess.run(ltn.Literal(True, B, c).tensor)

d_features = sess.run(d.tensor)[0]
print "the features of d are", d_features
print "d is of type A with degree =", sess.run(ltn.Literal(True, A, d).tensor)
print "d is of type B with degree =", sess.run(ltn.Literal(True, B, d).tensor)


As = np.squeeze(sess.run(A.tensor(),{person.tensor:data}))
Bs = np.squeeze(sess.run(B.tensor(),{person.tensor:data}))
Rs = np.squeeze(sess.run(R.tensor(),{couple.tensor:data_pairs}))

bestRs = argsort(-Rs)

fig = plt.figure(figsize=(16,10))


xs = data[:,0]
ys = data[:,1]
zs = data[:,2]

ax = fig.add_subplot(231,projection='3d')
for f in [a1_features,a2_features,a3_features]:
    yg = ax.scatter(f[0],f[1],f[2], marker='o',color="red")

for f in [b1_features,b2_features,b3_features]:
    yg = ax.scatter(f[0],f[1],f[2], marker='o',color="blue")

colors = cm.jet(As)
colmap = cm.ScalarMappable(cmap=cm.jet)
colmap.set_array(As)
ax = fig.add_subplot(232,projection='3d')
yg = ax.scatter(xs, ys, zs, c=colors, marker='o')
# cb = fig.colorbar(colmap)

colors = cm.jet(Bs)
colmap = cm.ScalarMappable(cmap=cm.jet)
colmap.set_array(Bs)
ax = fig.add_subplot(235,projection='3d')
yg = ax.scatter(xs, ys, zs, c=colors, marker='o')
# cb = fig.colorbar(colmap)


ax = fig.add_subplot(233,projection='3d')

for i in range(len(data_pairs)):
    j = bestRs[i]
    yg = ax.plot([data_pairs[j,0],data_pairs[j,3]],
                 [data_pairs[j,1],data_pairs[j,4]],
                 [data_pairs[j,2],data_pairs[j,5]],c="red",lw=2*Rs[j])

ax = fig.add_subplot(236,projection='3d')
yg = ax.scatter(c_features[0],
                c_features[1],
                c_features[2], marker='o',color="orange")
yg = ax.scatter(d_features[0],
                d_features[1],
                d_features[2], marker='o',color="green")
yg = ax.scatter([1,0,0,1],[1,0,1,0],[1,1,0,0], marker='o',color="white")
plt.show()


sess.close()
