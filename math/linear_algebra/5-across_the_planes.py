#!/usr/bin/env python3
'''Across The Planes Function'''


def add_matrices2D(mat1, mat2):
    '''Across The Planes Function'''
    rows1 = len(mat1)
    rows2 = len(mat2)
    columns1 = len(mat1[0])
    columns2 = len(mat2[0])
    if rows1 != rows2 or columns1 != columns2:
        return None
    result = []
    for i in range(len(mat1)):
        sum_row = []
        for j in range(0, len(mat1[0])):
            sum_row.append(mat1[i][j]+mat2[i][j])
        result.append(sum_row)
    return result


mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
add_matrices2D(mat1, mat2)
add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]])
