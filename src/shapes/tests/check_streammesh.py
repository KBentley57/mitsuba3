"""Run the SGX pointer-interface regression against a configured Ninja build.

Usage: python src/shapes/tests/check_streammesh.py /path/to/build
Load the same compiler/dependency modules used to configure that build first.
"""
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    build = Path(sys.argv[1]).resolve()
    commands = subprocess.check_output(
        ['ninja', '-C', str(build), '-t', 'commands', 'streammesh'], text=True
    ).splitlines()
    original = shlex.split(next(c for c in commands
                               if '-c ' in c and 'src/shapes/streammesh.cpp' in c))
    # Match every ABI-affecting option, including -march (Dr.Jit alignment).
    command = [original[0]]
    args = iter(original[1:])
    for arg in args:
        if arg in ('-o', '-c', '-MF', '-MT'):
            next(args)
        elif arg != '-MD':
            command.append(arg)
    output = build / 'test_streammesh'
    command += [str(Path(__file__).with_name('streammesh_native.cpp').resolve()),
                '-L' + str(build), '-lmitsuba', '-Wl,-rpath,' + str(build),
                '-o', str(output)]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(build) + ':' + env.get('LD_LIBRARY_PATH', '')
    subprocess.run(command, cwd=build, env=env, check=True)
    subprocess.run([str(output), str(build)], env=env, check=True)


if __name__ == '__main__':
    main()
