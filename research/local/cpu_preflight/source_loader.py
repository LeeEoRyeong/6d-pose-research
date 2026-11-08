"""프로젝트 전체 import 없이 원본 함수 본문을 읽는 CPU 검증용 로더."""
import ast
import hashlib
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = {}


def source(path):
    if "!" in path:
        archive, member = path.split("!", 1)
        with zipfile.ZipFile(ROOT / archive) as z:
            raw = z.read(member)
    else:
        raw = (ROOT / path).read_bytes()
    EVIDENCE[path] = {"sha256": hashlib.sha256(raw).hexdigest()}
    return raw.decode("utf-8-sig")


def node_at(tree, dotted):
    node = tree
    for name in dotted.split("."):
        node = next(n for n in node.body if isinstance(
            n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    return node


def function(path, dotted, namespace):
    """원본 함수의 본문·인자 유지. 전체 모듈 import/클래스 생성은 하지 않는다."""
    tree = ast.parse(source(path), filename=path)
    node = node_at(tree, dotted)
    if node.decorator_list:
        raise RuntimeError(f"예상하지 않은 decorator: {path}:{dotted}")
    EVIDENCE[path].setdefault("functions", {})[dotted] = node.lineno
    module = ast.Module(body=[node], type_ignores=[])
    env = dict(namespace)
    exec(compile(module, path, "exec"), env)
    return env[node.name]


def prediction_geometry(np, cv2):
    """Tester.predict의 resize/pad/K 갱신 구간만 원문 AST로 실행한다."""
    path = "HccePose/tester.py"
    tree = ast.parse(source(path), filename=path)
    body = node_at(tree, "Tester.predict").body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "(height, width)")
    stop = next(i for i in range(start, len(body))
                if isinstance(body[i], ast.Assign)
                and ast.unparse(body[i].targets[0]) == "img_torch")
    nodes = body[start:stop]
    # 경계가 변경되면 GPU 구간을 실행하지 않고 중단한다.
    banned = {"torch", "self", "YOLO", "subprocess", "os"}
    if any(isinstance(n, ast.Name) and n.id in banned
           for stmt in nodes for n in ast.walk(stmt)):
        raise RuntimeError("전처리 AST 경계가 변경됨: 수동 검토 필요")
    EVIDENCE[path]["geometry_lines"] = [nodes[0].lineno, nodes[-1].end_lineno]

    def run(img, cam_K):
        env = {"np": np, "cv2": cv2, "img": img, "cam_K": cam_K,
               "depth_for_foundationpose": None}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), env)
        return env["img"], env["cam_K"], env["ratio_"], env["w_move"], env["h_move"]
    return run


def bop_metrics(np, spatial):
    misc_path = "bop_toolkit.zip!bop_toolkit/bop_toolkit_lib/misc.py"
    pose_path = "bop_toolkit.zip!bop_toolkit/bop_toolkit_lib/pose_error.py"
    transform = function(misc_path, "transform_pts_Rt", {"np": np})
    env = {"np": np, "spatial": spatial,
           "misc": types.SimpleNamespace(transform_pts_Rt=transform)}
    return types.SimpleNamespace(
        add=function(pose_path, "add", env),
        adi=function(pose_path, "adi", env))
