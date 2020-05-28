#pragma once

#include "matrix.cpp"

template <class val_t>
class Activation
{
	typedef unsigned dim_t;
	std::string _type;

public:
	Activation()
	{
		_type = "sigmoid";
	}
	Activation(std::string activation)
	{
		_type = activation;
	}

	/**
	 * Activate
	 * @param in Matrix* to do math on
	 * @return Matrix* A new matrix with result of function
	 */
	Matrix<val_t> operator()(const Matrix<val_t> &in) const
	{
		Matrix<val_t> ret(in.h(), in.w());
		for (dim_t i = 0; i < in.h(); ++i)
			for (dim_t j = 0; j < in.w(); ++j)
			{
				val_t t = NULL;
				t = 1 / (1 + (std::exp(-1 * in.get(i, j))));
				ret.set(i, j, t);
			}
		return ret;
	}
	// overload for inplace calculation (non const)
	Matrix<val_t> operator()(Matrix<val_t> &in) const
	{
		for (dim_t i = 0; i < in.h(); ++i)
			for (dim_t j = 0; j < in.w(); ++j)
			{
				val_t t = INFINITY;
				t = 1 / (1 + (std::exp(-1 * in.get(i, j))));
				in.set(i, j, t);
			}
		return in;
	}

	Matrix<val_t> deriv(const Matrix<val_t> &in) const
	{
		Matrix<val_t> ret(in.h(), in.w());
		for (dim_t i = 0; i < in.h(); ++i)
			for (dim_t j = 0; j < in.w(); ++j)
			{
				val_t t = in.get(i, j) * (1 - in.get(i, j));
				ret.set(i, j, t);
			}
		return ret;
	}
};

