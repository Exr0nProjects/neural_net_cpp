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
                // if (_type == "sigmoid")
                // {
                t = 1 / (1 + (std::exp(-1 * in.get(i, j))));
                // }
                ret.set(i, j, t);
            }
        return ret;
    }
    // overload for inplace calculation (non const)
    Matrix<val_t> operator()(Matrix<val_t> &in) const
    {
        // return in;
        for (dim_t i = 0; i < in.h(); ++i)
            for (dim_t j = 0; j < in.w(); ++j)
            {
                val_t t = INFINITY;
                // if (strcmp(_type.c_str(), "sigmoid") == 0) // TODO: causes segfault
                // {
                t = 1 / (1 + (std::exp(-1 * in.get(i, j))));
                // }
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
                val_t t = NULL;
                t = in.get(i, j) * (1 - in.get(i, j));
                ret.set(i, j, t);
            }
        return ret;
    }
};
#pragma once

#include "matrix.cpp"
#include "activation.cpp"

#include <string>
#include <cmath>
#include <random>
#include <exception>

/**
 * Layer : Abstracted math
 */
template <class val_t>
class Layer
{
    typedef unsigned dim_t;
    Matrix<val_t> _syn;
    Activation<val_t> *_actv;
    dim_t _width, _height;

public:
    /**
   * Default constructor
   */
    Layer()
    {
        _syn = Matrix<val_t>(1, 1);
        _actv = new Activation<val_t>("sigmoid");
        _width = 0;
        _height = 0;
    }
    /**
   * "Basic" constructor
   * @param in_size Dimension of input to this layer
   * @param out_size Dimension of expected output (defaults to in_size)
   */
    Layer(const dim_t in_size, const dim_t out_size, const unsigned seed = 1) : _height(in_size), _width(out_size), _syn(in_size, out_size, 1)
    {
        _actv = new Activation<val_t>("sigmoid");
    }

    /**
   * "Activation" constructor
   * @param in_size Dimension of input to this layer
   * @param out_size Dimension of expected output
   * @param activation Activation function to use
   */
    Layer(const dim_t in_size, const dim_t out_size, const std::string activation, const unsigned seed = 1) : Layer(in_size, out_size, seed)
    {
        _actv = new Activation<val_t>(activation);
    }

    /**
   * Copy constructor
   * @param src The Layer to copy from
   */
    Layer(const Layer &src) : _syn(src.syn_raw())
    {
        _width = src.out_size();
        _height = src.in_size();
        _actv = new Activation<val_t>(*(src.actv_raw()));
    }

    ~Layer()
    {
        delete _actv;
    }

    /* methods */
    Layer &operator=(const Layer &o)
    {
        delete this;
        this = new Layer(o);
        return *this;
    }

    // TODO
    // template<typename... Args> // https://stackoverflow.com/a/16338804
    // void forEach(std::function<void()> const& lambda, Args... args)

    // getters
    dim_t in_size() const { return _height; }
    dim_t out_size() const { return _width; }
    const Matrix<val_t> &syn_raw() const { return _syn; }
    const Activation<val_t> *const actv_raw() const { return _actv; }

    // setters
    void update_raw(const Matrix<val_t> &mod)
    {
        if (mod.w() != _width || mod.h() != _height)
            throw std::domain_error("Invalid matrix dimensions to update layer!");
        for (dim_t i = 0; i < _height; ++i)
            for (dim_t j = 0; j < _width; ++j)
            {
                _syn.set(i, j, _syn.get(i, j) + mod.get(i, j));
            }
    }

    /**
   * Feed - feed forward through this layer
   * @param in Matrix of height in_size
   * @return Matrix of height out_size
   */
    Matrix<val_t> feed(const Matrix<val_t> &in) const
    {
        if (in.w() != _height)
            throw std::domain_error("Invalid input matrix width for network propogation!");


    Matrix<val_t> p = Matrix<val_t>::dot(in, _syn);
        (*_actv)(p); // TODO: copy construction involved in this line somewhere
        return p;
    }

    /**
   * Calculates and applies the weight changes asserted by backpropogation
   *
   * @param inp Weights that were input to this layer during training
   * @param out Weights that this layer returned
   * @param err The error in the returned value `out`
   * @return The modifications made
   */
    Matrix<val_t> backprop(const Matrix<val_t> &inp, const Matrix<val_t> &out, Matrix<val_t> err)
    {
        if (inp.h() != err.h())
            throw std::domain_error("Invalid 'inp' matrix dimensions for back propogation!");
        if (out.w() != err.w() || out.h() != err.h())
            throw std::domain_error("Invalid 'out' or 'err' matrix dimensions for back propogation!");

        err *= (_actv->deriv(out));
        update_raw(Matrix<val_t>::dot(Matrix<val_t>::transpose(inp), err));

        return err;
    }
};
#pragma once

#include <vector>
#include <cmath>
#include <iomanip>

/**
 * Matrix: Mathamatical Workhorse
 * Designed to do all of the heavy lifting math wise for the network
 * Will be modified to work with CUDA later
 */
template <class val_t>
class Matrix
{
    typedef unsigned dim_t;
    val_t **_data;
    dim_t _height;
    dim_t _width;
    int _id;

    static val_t **allocDims(dim_t h, dim_t w)
    {
        val_t **ret = new val_t *[h];
        for (dim_t i = 0; i < h; ++i)
        {
            ret[i] = new val_t[w];
        }
        return ret;
    }

public:
    /* static */
    /**
   * Return a new matrix that is the transposition of this matrix
   * @param src The source matrix (unmodified)
   * @return The transposed matrix
   */
    static Matrix transpose(const Matrix &src)
    {
        Matrix ret(src.w(), src.h());
        for (dim_t h = 0; h < src.h(); ++h)
            for (dim_t w = 0; w < src.w(); ++w)
                ret.set(w, h, src.get(h, w));
        return ret;
    }

    /**
   * Return a new matrix that is this matrix multiplied by another
   * @param lhs Matrix to be multiplied on
   * @param rhs Matrix to be multiplied by
   * @return The product matrix
   */
    static Matrix dot(const Matrix &lhs, const Matrix &rhs)
    {
        if (lhs.w() != rhs.h())
            throw std::domain_error("Invalid matrix dimensions for static dot multiplication!");
        Matrix ret(lhs.h(), rhs.w());
        for (dim_t r = 0; r < lhs.h(); ++r)
        {
            for (dim_t c = 0; c < rhs.w(); ++c)
            {
                val_t sum = 0;
                for (dim_t i = 0; i < lhs.w(); ++i)
                {
                    sum += lhs.get(r, i) * rhs.get(i, c);
                }
                ret.set(r, c, sum);
            }
        }
        return ret;
    }

    /* constructors/destructors */
    Matrix()
    {
        _data = new val_t *;
        *_data = new val_t;
        _height = 0;
        _width = 0;
        _id = rand();
    }

    /**
   * Empty Initializer Constructor - Create a new emtpy Matrix of w by h
   * @param height Height of the new Matrix
   * @param width Width of the new Matrix
   */
    Matrix(const dim_t height, const dim_t width, const int random = 0) : _height(height), _width(width)
    {
        if (random)
            std::srand(random);
        _data = allocDims(_height, _width);
        for (dim_t i = 0; i < _height; ++i)
        {
            for (dim_t j = 0; j < _width; ++j)
            {
                if (random)
                    _data[i][j] = ((val_t)std::rand() / RAND_MAX * 2 - 1);
                else
                    _data[i][j] = 0;
            }
        }
    }

    /**
   * Initalizer Constructor - Create a Matrix from a 2D vector
   * @param src 2D vector to copy from
   */
    Matrix(const std::vector<std::vector<val_t>> &src) : Matrix(src.size(), src[0].size())
    {
        // copy data
        for (dim_t i = 0; i < _height; ++i)
        {
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
    Matrix(const Matrix &src) : Matrix(src.height(), src.width())
    {
        for (dim_t i = 0; i < _height; ++i)
            for (dim_t j = 0; j < _width; ++j)
                set(i, j, src.get(i, j));
    }

    ~Matrix()
    {
        if (_data == nullptr)
            return;
        for (dim_t i = 0; i < _height; ++i)
        {
            if (_data[i] != nullptr)
            {
                delete[] _data[i];
            }
        }
        delete[] _data;
    }

    /* methods */
    Matrix &operator=(const Matrix &o)
    { // TODO: rewrite for efficiency? https://docs.microsoft.com/en-us/archive/msdn-magazine/2005/september/c-at-work-copy-constructors-assignment-operators-and-more
        if (this == &o)
            return *this; // otherwise "heap will get corrupted instantly" pg 10 of (http://www.umich.edu/~eecs381/lecture/Objectdynmemory.pdf)
        this->~Matrix();
        _height = o.h();
        _width = o.w();
        _data = allocDims(o.h(), o.w());
        for (dim_t i = 0; i < o.h(); ++i)
        {
            for (dim_t j = 0; j < o.h(); ++j)
            {
                _data[i][j] = o.get(i, j);
            }
        }
        return *this;
    }

    /**
   * Prints matrix to stdout
   * @param precision Sets precision of output, number of sigfigs
   */
    void print(const unsigned precision = 3) const
    {
        //printf("Printing Matrix %x\n", this);

        for (int i = 0; i < _height; ++i)
        {
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
    inline val_t get(const dim_t y, const dim_t x) const { return _data[y][x]; }
    /**
   * Set the value in the matrix at (x, y)
   * @param y The row of the element to be set
   * @param x The column of the element to be set
   * @param dat The data to be copied into that position
   */
    inline void set(const dim_t y, const dim_t x, const val_t &dat)
    {
        _data[y][x] = dat;
    }

    // TODO: check that this works
    void forEach(std::function<void(dim_t, dim_t, Matrix<val_t> *const, Matrix<val_t> *const)> const &lambda)
    {
        for (dim_t i = 0; i < h(); ++i)
            for (dim_t j = 0; j < w(); ++j)
                lambda(i, j, this);
    }

    /**
   * Operator overloads
   */
    Matrix<val_t> operator-=(const Matrix<val_t> &o)
    {
        if (w() != o.w() || h() != o.h())
            throw std::domain_error("Invalid matrix dimesions for element-wise subtract!");
        for (dim_t i = 0; i < h(); ++i)
            for (dim_t j = 0; j < w(); ++j)
                set(i, j, get(i, j) - o.get(i, j));
	return *this;
    }
    Matrix<val_t> operator*=(const Matrix<val_t> &o)
    {
        if (w() != o.w() || h() != o.h())
            throw std::domain_error("Invalid matrix dimesions for element-wise multiply!");
        for (dim_t i = 0; i < h(); ++i)
            for (dim_t j = 0; j < w(); ++j)
                set(i, j, get(i, j) * o.get(i, j));
        return *this;
    }

    /**
   * Return a pointer to a new matrix that is the transposition of this matrix
   * @return Matrix* The transposed matrix
   * @deprecated
   */
    Matrix &transpose() const
    {
        for (dim_t h = 0; h < _height; ++h)
            for (dim_t w = h + 1; w < _width; ++w)
            {
                val_t t = get(h, w);
                set(h, w, get(w, h));
                set(w, h, t);
            }
        return *this;
    }

    /**
   * Take the `e`th power of each value in the matrix
   * @return Matrix* The exponentiated matrix
   * @deprecated
   */
    Matrix<val_t> exp() const
    {
        Matrix ret(_height, _width);
        for (dim_t i = 0; i < _height; ++i)
            for (dim_t j = 0; j < _width; ++j)
            {
                ret.set(i, j, (val_t)std::exp(get(i, j)));
            }
        return ret;
    }
};
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
#pragma once

#include <vector>

#include "matrix.cpp"
#include "layer.cpp"
#include "activation.cpp"
#include "utility.cpp"

/**
 * Network - a general model framework
 */
class Network
{
    typedef float val_t;
    typedef unsigned dim_t;

    std::vector<Layer<val_t>> layers;
    unsigned _epoch_size;
    unsigned _status_log_count;
    dim_t _input_dim;

public:
    /**
   * Default constructor
   */
    Network() {}
    /**
   * "Normal" constructor
   * @param input_dim Dimensionality of the input vector
   */
    Network(dim_t input_dim)
    {
        _input_dim = input_dim;
    }

    /**
   * Adds a layer to the model
   * @param out_dim Dimensionality of the output of this layer
   */
    void addLayer(dim_t out_dim)
    {
        dim_t prev = _input_dim;
        if (layers.size())
            prev = layers[layers.size() - 1].out_size();
        Layer<val_t> nxt(prev, out_dim);
        layers.push_back(nxt);
    }

    /**
   * Feed an input vector through the network to check the result
   */
    Matrix<val_t> feed(Matrix<val_t> inp)
    {
        for (const Layer<val_t> &layer : layers)
        {
            inp = layer.feed(inp);
        }
        return inp;
    }

    /**
   * Train the network
   * @param epoch_size How many iterations to train through
   */
    void train(const Matrix<val_t> &inp, Matrix<val_t> exp, const unsigned epoch = 500000, const unsigned updates = 100)
    {
        printf("\n\n");
        for (unsigned i = 0; i < epoch; ++i)
        {
            std::vector<Matrix<val_t>> snapshots;
            snapshots.push_back(inp);

            // Feed forward
            for (const Layer<val_t> &layer : layers)
            {
                snapshots.push_back(layer.feed(snapshots[snapshots.size() - 1]));
            }

            Matrix<val_t> error = exp;
            error -= snapshots[snapshots.size() - 1];

            if (i % (epoch / 100) == 0)
            {
                // printf("Input:\n");
                // inp.print();
                // printf("Output:\n");
                // snapshots[snapshots.size()-1].print();
                // printf("Exepected:\n");
                // exp.print();

                progressBar(50, (double)i / epoch, 2, "=", " ", 10);

                val_t average_error = 0;
                for (int i = 0; i < error.h(); ++i)
                    average_error += abs(error.get(i, 0));
                average_error /= exp.h() * exp.w();
                printf("error = %.5f\n", average_error);
            }

            //printf("starting backprop...\n\n");

            // Backprop
            for (int i = layers.size() - 1; i >= 0; --i)
            {
                //printf("backprop i=%d\n", i);
                Matrix<val_t> delta = layers[i].backprop(snapshots[i], snapshots[i + 1], error);  //  TODO: don't  copy exp then copy back, just use the same memory
                //printf("    got delta\n");
                Matrix<val_t> synT = Matrix<val_t>::transpose(layers[i].syn_raw());
                //printf("    got syn transpose\n");

                error = Matrix<val_t>::dot(delta, synT);
                //Matrix<val_t>::dot(delta, synT);
                //printf("    got error at %x\n", &error);
            }
        }
    }
};
#include <cstdio>
#include <chrono>
#include <stdexcept>

void progressBar(int width, double progress, int overwrite = true, const std::string &fill = "=", const std::string &empty = " ", int segment_size = 4)
{
    if (progress > 1 || progress < 0)
        throw std::domain_error("Invalid progress!");

    for (int i = 0; i < overwrite; ++i)
        printf("\033[F");

    int pos = width * progress;
    if (segment_size > pos)
        segment_size = pos + 1; // make gaps at the beginning
    static int dot_offset = 0;
    dot_offset = (dot_offset + 1) % segment_size;

    printf("[");
    if (progress * 100 + 1 < 100)
    {
        for (int i = 0; i < width; ++i)
        {
            if (i < pos)
            {
                if (i % segment_size == dot_offset)
                    printf("%s", empty.c_str());
                else
                    printf("%s", fill.c_str());
            }
            else if (i == pos)
                printf(">");
            else if (i > pos)
                printf(" ");
        }
    }
    else
    {
        for (int i = 0; i < width; ++i)
            printf("%s", fill.c_str());
    }
    printf("] %3d%%\n", (int)(progress * 100 + 1));
}
