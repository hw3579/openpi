
# 方案 A：按变量名排行（只统计可达变量）

def top_memory_bindings(top_n=30, module_prefixes=None, include_frames=True, use_asizeof=True):
    """
    收集当前进程里“能拿到名字的对象”，按占用内存排序。
    - module_prefixes: 只统计这些前缀的模块（列表，如 ['myproj', 'package_a']）。
      为空则不筛选（会扫到很多第三方库，全量可能很大）。
    - include_frames: 是否把所有活动栈帧的 locals/globals 也并入统计。
    - use_asizeof: 有安装 pympler 时更准确；否则退回 getsizeof（低估容器）。
    """
    import sys, inspect, types, gc

    # 选择计量函数
    if use_asizeof:
        try:
            from pympler import asizeof
            sizeof = asizeof.asizeof
        except Exception:
            import sys as _sys
            sizeof = _sys.getsizeof
    else:
        import sys as _sys
        sizeof = _sys.getsizeof

    seen = set()  # 已计量的对象 id，避免重复
    rows = []     # (size, typename, where, varname, obj_id)

    def record(name, obj, where):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        try:
            sz = int(sizeof(obj))
        except Exception as e:
            # 保险：计量失败就记 0
            sz = 0
        tname = type(obj).__name__
        rows.append((sz, tname, where, name, oid))

    # 1) 栈帧里的 locals/globals（通常包含你当前在调试的变量）
    if include_frames:
        for f in inspect.stack():
            frame = f.frame
            # locals
            for k, v in list(frame.f_locals.items()):
                record(k, v, f"frame:{frame.f_code.co_filename}:{frame.f_lineno}")
            # globals
            for k, v in list(frame.f_globals.items()):
                record(k, v, f"globals:{frame.f_code.co_filename}")

    # 2) 指定模块的 __dict__（函数外的全局变量通常在这里）
    import sys as _sys
    for mod_name, mod in list(_sys.modules.items()):
        if not mod or not hasattr(mod, "__dict__"):
            continue
        if module_prefixes:
            if not any(mod_name == p or mod_name.startswith(p + ".") for p in module_prefixes):
                continue
        # 记录模块全局变量
        where = f"module:{mod_name}"
        for k, v in list(mod.__dict__.items()):
            # 跳过模块对象自身，避免爆炸
            if isinstance(v, types.ModuleType):
                continue
            record(k, v, where)

    # 排序并输出
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:top_n]

def print_top_memory_bindings(top_n=30, module_prefixes=None, include_frames=True, use_asizeof=True):
    rows = top_memory_bindings(top_n=top_n,
                               module_prefixes=module_prefixes,
                               include_frames=include_frames,
                               use_asizeof=use_asizeof)
    # 故意不打印 repr()，避免触发对象副作用
    print(f"{'SIZE(bytes)':>12}  {'TYPE':<24}  {'WHERE':<40}  NAME")
    for sz, tname, where, name, oid in rows:
        print(f"{sz:12d}  {tname:<24}  {where:<40}  {name}")


###### 
# 方案 B：按全进程对象排行（不需要变量名）

def top_memory_objects(top_n=30, use_asizeof=True):
    import gc, sys
    if use_asizeof:
        try:
            from pympler import asizeof
            sizeof = asizeof.asizeof
        except Exception:
            sizeof = sys.getsizeof
    else:
        sizeof = sys.getsizeof

    objs = gc.get_objects()
    rows = []
    seen = set()
    for o in objs:
        oid = id(o)
        if oid in seen:
            continue
        seen.add(oid)
        try:
            sz = int(sizeof(o))
        except Exception:
            sz = 0
        rows.append((sz, type(o).__name__, oid))
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:top_n]

def print_top_memory_objects(top_n=30, use_asizeof=True):
    rows = top_memory_objects(top_n=top_n, use_asizeof=use_asizeof)
    print(f"{'SIZE(bytes)':>12}  {'TYPE':<24}  OBJECT_ID")
    for sz, tname, oid in rows:
        print(f"{sz:12d}  {tname:<24}  0x{oid:x}")
