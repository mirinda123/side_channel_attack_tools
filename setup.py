# 打包成模块压缩包
# from setuptools import setup, find_packages
# setup(
#     name="side_channel_attack",  # 包名
#     version="0.1",  # 版本
#     # 最重要的就是py_modules和packages
#     py_modules=["side_channel_attack.attack_method","side_channel_attack.base_trace","side_channel_attack.preprocessing","side_channel_attack.gpu_sca"],  # py_modules : 打包的.py文件
#     packages=["major.major1"],  # packages: 打包的python文件夹
#     # keywords=("AI", "Algorithm"),  # 程序的关键字列表
#     description="side channel attack tools",                 # 简单描述
#     long_description="a deeper wider tool for side channel attack", # 详细描述
#     # license="MIT Licence",  # 授权信息
#     url="https://blog.csdn.net/qq_41375318/article/details/115568470",  # 官网地址
#     author="yang su",  # 作者
#     author_email="1171657161@qq.com",  # 作者邮箱
#     # packages=find_packages(), # 需要处理的包目录（包含__init__.py的文件夹）
#     # platforms="any",  # 适用的软件平台列表
#     # install_requires=[],  # 需要安装的依赖包
#     # 项目里会有一些非py文件,比如html和js等,这时候就要靠include_package_data和package_data来指定了。
#     # scripts=[],  # 安装时需要执行的脚本列表
#     # entry_points={     # 动态发现服务和插件
#     #     'console_scripts': [
#     #         'jsuniv_sllab = jsuniv_sllab.help:main'
#     #     ]
#     # }
#
# )
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="side_channel_attack",
    version="0.0.1",
    author="su yang",
    author_email="1152036203@qq.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "side_channel_attack"},
    packages=setuptools.find_packages(where="side_channel_attack"),
    python_requires=">=3.6",
)