**Using Theano with Nvidia GPUs:**

Assuming that you have built Theano from source with GPU acceleration enabled, you can now use it with a CUDA-enabled GPU as shown below:

Simply run:

    python gputest.py

In the source directory in the repository. 

You can also benchmark how long this will take by running:

    time python gputest.py

Use the Theano flag `device=cuda` to require the use of the GPU. Use the flag `device=cuda{0,1,...}` to specify which GPU to use.

A sample `~/.theanorc` configuration file is shown below, adapt to your needs:

    [global]
    floatX = float32
    device = cuda
    optimizer = fast_run
    
    [lib]
    cnmem = 0.9
    
    [nvcc]
    fastmath = True
    
    [blas]
    ldflags = -llapack -lblas


We will cover what these options mean in a dedicated section below.

The program `testgpu.py` just computes `exp()` of a bunch of random numbers. The program uses the `theano.shared()` function to make sure that the input `x` is stored on the GPU.

To test this on a CPU and to see how long this would take without any acceleration, run:

    $ THEANO_FLAGS=device=cpu gputest.py
    [Elemwise{exp,no_inplace}(<TensorType(float64, vector)>)]
    Looping 1000 times took 2.271284 seconds
    Result is [ 1.23178032  1.61879341  1.52278065 ...,  2.20771815  2.29967753
      1.62323285]
    Used the cpu

To test the same via a GPU accelerated context under [libgpuarray](http://deeplearning.net/software/libgpuarray/installation.html), run:

    $ THEANO_FLAGS=device=cuda0 gputest.py
    Using cuDNN version 5110 on context None
    Mapped name None to device cuda: GeForce GTX 1060 (0000:01:00.0)
    [GpuElemwise{exp,no_inplace}(<GpuArrayType<None>(float32, (False,))>), HostFromGpu(gpuarray)(GpuElemwise{exp,no_inplace}.0)]
    Looping 1000 times took 0.190633 seconds
    Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
      1.62323296]
    Used the gpu


**Returning handles to Device-Allocated Data:**

By default, functions that execute on the GPU still return a standard [numpy](http://www.numpy.org/) [ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html). A transfer operation is inserted just before the results are returned to ensure stateful consistency with CPU (host) code. This allows changing the device handle contexts by modifying a runtime flag without altering the host code.

Theano can also return the GPU object directly, albeit at a loss of flexibility. The program `gpuobject.py` does that:

    python gpuobject.py

Here, `tensor.exp(x).transfer(None)` means “copy `exp(x)` to the GPU”, with None the default GPU context when not explicitly given.

The output is:

    $ THEANO_FLAGS=device=cuda0 gpuobject.py
    Using cuDNN version 5110 on context None
    Mapped name None to device cuda: GeForce GTX 1060 (0000:01:00.0)
    [GpuElemwise{exp,no_inplace}(<GpuArrayType<None>(float32, (False,))>)]
    Looping 1000 times took 0.012451 seconds
    Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
      1.62323296]
    Used the gpu


While the time per call appears to be much lower than the two previous invocations (and should indeed be lower, since we avoid a transfer), the massive speedup we obtained is in part due to the [asynchronous nature of execution on GPUs](https://www.linkedin.com/pulse/directx-12-demystifying-asynchronous-compute-nvidia-amd-dennis-mungai).

The object returned is a `GpuArray` from `pygpu`. It mostly acts as a numpy ndarray with some exceptions due to its data being on the GPU. You can copy it to the host and convert it to a regular ndarray by using usual numpy casting such as `numpy.asarray()`.


**What Can be Accelerated on the GPU:**

The performance characteristics will of course vary from device to device, and also as we refine our implementation:

  1. In general, [matrix multiplication](https://stuff.mit.edu/afs/sipb/project/www/matlab/imatlab/node10.html), [convolution](http://www.dspguide.com/ch6.htm), and [large element-wise operations](https://en.wikipedia.org/wiki/Matrix_multiplication) can be accelerated a lot (5-50x) *when arguments are large enough to saturate the host processor*.
   
  2. Indexing, dimension-shuffling and constant-time reshaping will be equally fast on GPU as on CPU.
   
  3. Note that summation over rows/columns of tensors can be a little slower on the GPU than on the CPU, owing to a higher preemption penalty on most GPUs. On [Nvidia Pascal GPUs](https://en.wikipedia.org/wiki/Pascal_%28microarchitecture%29), where [instruction level fine-grained preemption](http://www.anandtech.com/show/10325/the-nvidia-geforce-gtx-1080-and-1070-founders-edition-review/10) is enabled at an architectural level, this limitation does not apply.
  
 However, note that copying large quantities of data to and from a device is relatively slow, and ***often cancels most of the advantage of one or two accelerated functions on that data. Getting GPU performance largely hinges on making data transfer to the device pay off.***

The libgpuarray backend supports all regular theano data types (float32, float64, int, ...), however GPU support varies and some units (particularly consumer-grade Nvidia GPUs) can’t deal with double (float64) or small (less than 32 bits like int16) data types (Addressed quite well on Pascal). You may get an error at compile time or runtime if this is the case.

By default, all inputs will get transferred to the GPU. You can prevent an input from getting transferred by setting its `tag.target` attribute to `‘cpu’`.


**Tips for Improving Performance on GPUs:**

Consider adding floatX=float32 (or the type you are using) to your .theanorc file if you plan to do a lot of GPU work.The GPU backend supports float64 variables, but they are still slower to compute than float32. The more float32, the better GPU performance you will get.
    
Prefer constructors like matrix, vector and scalar (which follow the type set in floatX) to dmatrix, dvector and dscalar. The latter enforce double precision (float64 on most machines), which slows down GPU computations on current hardware.

Minimize transfers to the GPU device by using shared variables to store frequently-accessed data (see shared()). When using the GPU, tensor shared variables are stored on the GPU by default to eliminate transfer time for GPU ops using those variables.

If you aren’t satisfied with the observed performance, run your script with the `profile=True` flag. This should print some timing information at program termination. To interprete: Is time being used sensibly? If an op or Apply is taking more time than its share, then if you know something about GPU programming, have a look at how it’s implemented in `theano.gpuarray`. Check the line similar to Spent `Xs(X%)` in cpu -bound operations, `Xs(X%)` in GPU operations and `Xs(X%)` in transfer operations. This can help the user quickly identify potential bottlenecking as it happens in real time.

To investigate whether all the operations in the computational graph are running on the GPU, it is possible to debug or check your code by providing a value to `assert_no_cpu_op` flag, i.e. `warn`, for warning, `raise` for raising an error or `pdb` for putting a breakpoint in the computational graph if there is a CPU-bound operation.
 
 Please note that `config.lib.cnmem` and `config.gpuarray.preallocate` controls GPU memory allocation when using the CUDA backend (now deprecated) and the GpuArray Backend (current) as theano backends respectively.

**GPU Asynchronous Capabilities:**

By default, all operations on the GPU are run asynchronously. This means that they are only scheduled to run and the function returns. This is made somewhat transparent by the underlying `libgpuarray`back-end.

A forced synchronization point is introduced when doing memory transfers between the device and the host.

It is possible to force synchronization for a particular GpuArray by calling its `sync()` method. This is useful to get accurate timings when doing benchmarks.

**Changing the Value of Shared Variables:**

To change the value of a shared variable, e.g. to provide new data to processes, use `shared_variable.set_value(new_value)` constructor as shown in the program lregress.py. 

**Testing your understanding:**

Modify and execute this program to run on the GPU with `floatX=float32` and time it using the command line time `lregress.py`. 

**Observations:**

1. Is there an increase in speed compared to running it on the CPU vs the GPU?

2. Where does it come from? (Use `profile=True` flag.)

3. What can be done to further increase the speed of the GPU version? Test your ideas.

**Solution:**

See `solution.py` for a proposed solution.

**Extra homework:**

Build both tensorflow and theano from source, with GPU acceleration enabled.
Also, ensure that they're fully configured to utilize the GPU on the target platform.

