# 使用脚本位置定位仓库，不依赖当前工作目录。
const cuda_root = path self ../hw3/.tools/cuda-13.0.2
if ($cuda_root | path join 'bin' 'nvcc' | path exists) {
    $env.CUDA_HOME = $cuda_root
    $env.CUDACXX = ($cuda_root | path join 'bin' 'nvcc')
    $env.PATH = ($env.PATH | prepend ($cuda_root | path join 'bin') | uniq)
}
