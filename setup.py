from setuptools import find_packages,setup

HYPEN_E_DOT = '-e .'
def get_requirements(filepath:str)->list[str]:
    '''this function will return list of requirements'''
    requirements = []
    with open(filepath) as ptr:
        requirements = ptr.readlines()
        requirements = [rqr.replace("\n"," ") for rqr in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        

setup(
    name = "ml-project",
    version='0.0.1',
    author='cherry',
    packages = find_packages(),
    install_requirements = get_requirements('requirements.txt')
)#metadata of the project