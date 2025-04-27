from g4f import Provider, models
print("Available Providers:", [p for p in dir(Provider) if not p.startswith('_')])
print("Available Models:", [m for m in dir(models) if not m.startswith('_')])