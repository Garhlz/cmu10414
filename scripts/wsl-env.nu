# 使用脚本位置定位仓库，不依赖当前工作目录。
def --env cmu10414-cuda-env [] {
    const cuda_root = path self ../hw3/.tools/cuda-13.0.2
    if ($cuda_root | path join 'bin' 'nvcc' | path exists) {
        $env.CUDA_HOME = $cuda_root
        $env.CUDACXX = ($cuda_root | path join 'bin' 'nvcc')
        $env.PATH = ($env.PATH | prepend ($cuda_root | path join 'bin') | uniq)
    }
}

cmu10414-cuda-env

# 本脚本供手动加载；持久配置应在 env.nu 的 mise 初始化之前设置 CUDA。
