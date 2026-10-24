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

# mise 的环境钩子可能重写 PATH；本脚本应在 use mise.nu 之后加载。
if not ($env.CMU10414_CUDA_HOOK? | default false) {
    $env.config.hooks.pre_prompt = ($env.config.hooks.pre_prompt | append { cmu10414-cuda-env })
    $env.config.hooks.env_change.PWD = ($env.config.hooks.env_change.PWD | default [] | append { cmu10414-cuda-env })
    $env.CMU10414_CUDA_HOOK = true
}
