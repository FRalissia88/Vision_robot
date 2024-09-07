#ifndef CV2_HPP
#define CV2_HPP

//warning number '5033' not a valid compiler warning in vc12
#if defined(_MSC_VER) && (_MSC_VER > 1800)
// eliminating duplicated round() declaration
#define HAVE_ROUND 1
#pragma warning(push)
#pragma warning(disable:5033)  // 'register' is no longer a supported storage class
#endif

// #define CVPY_DYNAMIC_INIT
// #define Py_DEBUG

#if defined(CVPY_DYNAMIC_INIT) && !defined(Py_DEBUG)
#   define Py_LIMITED_API 0x03030000
#endif

#include <cmath>
#include <Python.h>
#include <limits>

#if PY_MAJOR_VERSION < 3
#undef CVPY_DYNAMIC_INIT
#else
#define CV_PYTHON_3 1
#endif

#if defined(_MSC_VER) && (_MSC_VER > 1800)
#pragma warning(pop)
#endif

#define MODULESTR "cv2"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>

#include "pycompat.hpp"

class ArgInfo
{
private:
    static const uint32_t arg_outputarg_flag     = 0x1;
    static const uint32_t arg_arithm_op_src_flag = 0x2;

public:
    const char* name;
    bool outputarg;
    bool arithm_op_src;
    // more fields may be added if necessary

    ArgInfo(const char* name_, uint32_t arg_) :
        name(name_),
        outputarg((arg_ & arg_outputarg_flag) != 0),
        arithm_op_src((arg_ & arg_arithm_op_src_flag) != 0) {}

private:
    ArgInfo(const ArgInfo&) = delete;
    ArgInfo& operator=(const ArgInfo&) = delete;
};


#endif // CV2_HPP
