from pathlib import Path
from types import SimpleNamespace
import subprocess

from basketball_poc import analysis_launcher as launcher


def runtimes(tmp_path, monkeypatch):
    cpu = tmp_path/'.venv/Scripts/python.exe'
    gpu = tmp_path/'.venv-gpu/Scripts/python.exe'
    for file in (cpu,gpu):
        file.parent.mkdir(parents=True)
        file.touch()
    monkeypatch.setattr(launcher.sys, 'executable', str(cpu))
    return cpu.resolve(), gpu.resolve()


def test_auto_chooses_optional_healthy_gpu_runtime(tmp_path, monkeypatch):
    cpu,gpu = runtimes(tmp_path,monkeypatch)
    calls=[]
    def run(args,**kwargs):
        calls.append((args,kwargs))
        return SimpleNamespace(returncode=0,stdout='{"cuda_available":true,"torch":"test"}')
    monkeypatch.setattr(launcher.subprocess,'run',run)
    selected,evidence=launcher.select_python(tmp_path)
    assert selected==gpu and evidence['mode']=='auto'
    assert len(calls)==1 and 'shell' not in calls[0][1]


def test_broken_gpu_runtime_falls_back_to_cpu(tmp_path,monkeypatch):
    cpu,gpu=runtimes(tmp_path,monkeypatch)
    def run(args,**kwargs):
        return SimpleNamespace(returncode=1,stdout='') if args[0]==str(gpu) else SimpleNamespace(
            returncode=0,stdout='{"cuda_available":false,"torch":"cpu"}')
    monkeypatch.setattr(launcher.subprocess,'run',run)
    selected,evidence=launcher.select_python(tmp_path)
    assert selected==cpu and evidence['mode']=='auto-cpu'
    assert evidence['probes'][0]['healthy'] is False


def test_missing_optional_gpu_runtime_uses_current_cpu_without_probe_duplication(tmp_path,monkeypatch):
    cpu=tmp_path/'python.exe'
    cpu.touch()
    monkeypatch.setattr(launcher.sys,'executable',str(cpu))
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw: SimpleNamespace(
        returncode=0,stdout='{"cuda_available":false,"torch":"cpu"}'))
    assert launcher.select_python(tmp_path)[0]==cpu
    assert launcher.select_python(tmp_path,device='cpu')[1]['probes']==[]


def test_explicit_runtime_is_preserved(tmp_path):
    path=tmp_path/'user runtime.exe'
    path.touch()
    assert launcher.select_python(tmp_path,path)[0]==path


def test_healthy_gpu_build_without_cuda_prefers_installed_cpu_runtime(tmp_path,monkeypatch):
    cpu,gpu=runtimes(tmp_path,monkeypatch)
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw: SimpleNamespace(
        returncode=0,stdout='{"cuda_available":false,"torch":"test"}'))
    selected,evidence=launcher.select_python(tmp_path)
    assert selected==cpu and evidence['mode']=='auto-cpu'
