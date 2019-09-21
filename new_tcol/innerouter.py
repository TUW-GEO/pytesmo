# -*- coding: utf-8 -*-

'''
Module description
'''
# TODO:
#   (+) 
#---------
# NOTES:
#   -

from collections import  namedtuple

t_inner1=namedtuple('test1', ['t1', 't2'])
t_inner2=namedtuple('test2', ['t1', 't2'])

inner1 = t_inner1(1,2)
inner2 = t_inner2(1,2)

t_outer=namedtuple('OUTER', ['test1', 'test2'])
t_outer(inner1, inner2)