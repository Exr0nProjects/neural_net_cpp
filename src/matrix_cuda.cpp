#pragma once

#include <cmath>
#include <iomanip>
#include <vector>

#include <assert.h>

/**
 * Matrix: Mathamatical Workhorse
 * Designed to do all of the heavy lifting math wise for the network
 * Currently being modified to work with CUDA
 */
template <class val_t> class Matrix {
    typedef unsigned dim_t;
    val_t *_data;
    dim_t _height;
    dim_t _width;
    int _id;

    static val_t *allocDims(dim_t h, dim_t w) {
        val_t *ret = new val_t[h*w];
        return ret;
    }

    public:
    /* static */
    /**
     * Transposes this matrix in place
     * @param src The source matrix (modified)
     */
    static Matrix<val_t> transpose(Matrix &src) {
        for (dim_t h = 0; h < src.h(); ++h)
            for (dim_t w = 0; w < src.w(); ++w)
                src.set(w, h, src.get(h, w));
        return src;
    }

    /**
     * Return a new matrix that is this matrix multiplied by another
     * @param lhs Matrix to be multiplied on
     * @param rhs Matrix to be multiplied by
     * @return The product matrix
     */
    static Matrix dot(const Matrix &lhs, const Matrix &rhs) {
        if (lhs.w() != rhs.h())
            throw std::domain_error(
                    "Invalid matrix dimensions for static dot multiplication!");
        Matrix ret(lhs.h(), rhs.w());
        printf("        created ret\n");
        for (dim_t r = 0; r < lhs.h(); ++r) {
            for (dim_t c = 0; c < rhs.w(); ++c) {
                val_t sum = 0;
                for (dim_t i = 0; i < lhs.w(); ++i) {
                    sum += lhs.get(r, i) * rhs.get(i, c);
                }
                printf("            setting ret\n");
                ret.set(r, c, sum);
            }
        }
        printf("        about to return from matrix dot (ret at %x)\n", &ret);
        return ret;
    }

    /* constructors/destructors */
    Matrix() {
        _data = new val_t;
        _height = 0;
        _width = 0;
        _id = rand();
    }

    /**
     * Empty Initializer Constructor - Create a new emtpy Matrix of w by h
     * @param height Height of the new Matrix
     * @param width Width of the new Matrix
     */
    Matrix(const dim_t height, const dim_t width, const int random = 0)
        : _height(height), _width(width) {
            if (random)
                std::srand(random);
            _data = allocDims(_height, _width);
            for (dim_t i = 0; i < _height; ++i) {
                for (dim_t j = 0; j < _width; ++j) {
                    if (random)
                        set(i, j, (val_t)std::rand() / RAND_MAX * 2 - 1);
                    else
                        set(i, j, 0);
                }
            }
        }

    /**
     * Initalizer Constructor - Create a Matrix from a 2D vector
     * @param src 2D vector to copy from
     */
    Matrix(const std::vector<std::vector<val_t>> &src)
        : Matrix(src.size(), src[0].size()) {
            // copy data
            for (dim_t i = 0; i < _height; ++i) {
                // check shape
                if (src[i].size() != _width)
                    throw std::domain_error("Non rectangular vector input!");
                // copy row
                for (dim_t j = 0; j < _width; ++j)
                    set(i, j, src[i][j]);
            }
        }

    /**
     * Copy Constructor - Copy a matrix
     * @param src Matrix to copy from
     */
    Matrix(const Matrix &src) : Matrix(src.height(), src.width()) {
        //printf("ctor: copying matrix %x->%x\n", &src, this);
        for (dim_t i = 0; i < _height; ++i)
            for (dim_t j = 0; j < _width; ++j)
                set(i, j, src.get(i, j));
    }

    ~Matrix() {
        // TODO: rewrite for 1d array
        if (_data == nullptr)
            return;
        delete _data;
    }

    /* methods */
    Matrix &operator=(const Matrix &o) { // TODO: rewrite for efficiency?
      // https://docs.microsoft.com/en-us/archive/msdn-magazine/2005/september/c-at-work-copy-constructors-assignment-operators-and-more
      //printf("copy assignment of %x!\n", &o);
      if (this == &o)
        return *this; // otherwise "heap will get corrupted instantly" pg 10 of
      // (http://www.umich.edu/~eecs381/lecture/Objectdynmemory.pdf)
      printf("copy assignment called!");
      this->~Matrix();
      _height = o.h();
      _width = o.w();
      _data = allocDims(o.h(), o.w());
      for (dim_t i = 0; i < o.h(); ++i) {
        for (dim_t j = 0; j < o.h(); ++j) {
          set(i, j, o.get(i, j));
        }
      }
      return *this;
    }

    /**
     * Prints matrix to stdout
     * @param precision Sets precision of output, number of sigfigs
     */
    void print(const unsigned precision = 3) const {
        printf("Printing Matrix %x\n", this);

        for (int i = 0; i < _height; ++i) {
            for (int j = 0; j < _width; ++j)
                printf(" %+1.3f", get(i, j));
            printf("\n");
        }
        printf("\n");
    }

    /**
     * Returns the width of this matrix
     * @return dim_t The width of the matrix
     */
    inline dim_t width() const { return _width; };
    inline dim_t w() const { return width(); };

    /**
     * Returns the height of this matrix
     * @return dim_t The height of the matrix
     */
    inline dim_t height() const { return _height; };
    inline dim_t h() const { return height(); };

    /**
     * Get the value in the matrix at (x, y)
     * @param y The row of the element to be retrieved
     * @param x The column of the element to be retrieved
     * @return val_t The value of the element at that position
     */
    inline val_t get(const dim_t y, const dim_t x) const {
      return _data[y * _width + x];
    }
    /**
     * Set the value in the matrix at (x, y)
     * @param y The row of the element to be set
     * @param x The column of the element to be set
     * @param dat The data to be copied into that position
     */
    inline void set(const dim_t y, const dim_t x, const val_t &dat) {
        _data[y * _width + x] = dat;
    }

    // TODO: check that this works
    void forEach(std::function<void(dim_t, dim_t, Matrix<val_t> *const,
                Matrix<val_t> *const)> const &lambda) {
        for (dim_t i = 0; i < h(); ++i)
            for (dim_t j = 0; j < w(); ++j)
                lambda(i, j, this);
    }

    /**
     * Operator overloads
     */
    void operator-=(const Matrix<val_t> &o){
        if (w() != o.w() || h() != o.h())
            throw std::domain_error(
                    "Invalid matrix dimesions for element-wise subtract!");
        for (dim_t i = 0; i < h(); ++i)
            for (dim_t j = 0; j < w(); ++j)
                set(i, j, get(i, j) - o.get(i, j));
    }
    void operator*=(const Matrix<val_t> &o) {
        if (w() != o.w() || h() != o.h())
            throw std::domain_error(
                    "Invalid matrix dimesions for element-wise multiply!");
        for (dim_t i = 0; i < h(); ++i)
            for (dim_t j = 0; j < w(); ++j)
                set(i, j, get(i, j) * o.get(i, j));
    }
};
