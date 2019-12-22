from distutils.core import setup, Extension

module1 = Extension('gcophrotor',
          include_dirs = [#'/home/gabe/rosSrc/dependencies/gcop/lib/utils',
                          #'/home/gabe/rosSrc/dependencies/gcop/lib/algos',
                          #'/home/gabe/rosSrc/dependencies/gcop/lib/systems',
                          #'/home/gabe/rosSrc/dependencies/gcop/lib/systems/constraints',
                          #'/home/gabe/rosSrc/dependencies/gcop/lib/systems/costs',
                          #'/home/gabe/rosSrc/dependencies/gcop/lib/systems/manifolds',
                          '/usr/include/eigen3/',
                          '/usr/local/include/',
                          '/usr/local/include/gcop/',
                           ],
                          #'/home/gabe/rosSrc/dependencies/ceres-solver/include/ceres/internal/'],
          library_dirs = [#'/home/gabe/rosSrc/dependencies/gcop/build2/lib',
                          '/usr/local/lib','/usr/lib/x86_64-linux-gnu/'],
              #            '/home/gabe/rosSrc/dependencies/gcop/build2/lib/algos',
              #            '/home/gabe/rosSrc/dependencies/gcop/build2/lib/utils',
              #            '/home/gabe/rosSrc/dependencies/gcop/build2/lib/systems',
              #            '/home/gabe/rosSrc/dependencies/ceres-solver/build/lib'], 
          libraries = ['gcop_systems','gcop_algos','gcop_est','gcop_utils','gcop_views','glut','GLU',
                       'tinyxml','ceres','glog','casadi','g2o_incremental','g2o_solver_cholmod','gflags',
                       'cxsparse','tbbmalloc','tbb','cholmod','ccolamd','camd','colamd','amd','lapack',
                       'blas', 'suitesparseconfig', 'rt','cxsparse', 'gomp', 'pthread', 'spqr', 'dl'
                       ],
                      # ' libgcop_views.so.1.0.0',
                      # ' libgcop_systems.so',
          runtime_library_dirs = ['/usr/local/lib','/usr/lib/x86_64-linux-gnu/'],
          extra_compile_args=['-std=c++11'], sources = ['hrotorpython.cc'])

setup(name = 'PackageName', version = '1.0',
      description = 'GCOP for Hrotor Orange Picking',
      ext_modules = [module1])
