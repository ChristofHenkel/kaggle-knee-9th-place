from types import SimpleNamespace


class Config(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)

    def require(self, *names):
        missing = [name for name in names if not hasattr(self, name)]
        if missing:
            joined = ", ".join(missing)
            raise AttributeError(f"Missing required config value(s): {joined}")
        if len(names) == 1:
            return getattr(self, names[0])
        return tuple(getattr(self, name) for name in names)

    def to_dict(self):
        return dict(self.__dict__)

    def __str__(self):
        string = ["config"]
        string.append("=" * len(string[0]))
        if not self.__dict__:
            return "\n".join(string)
        longest_param_name = max([len(k) for k in [*self.__dict__]])
        for k, v in self.__dict__.items():
            string.append(f"{k.ljust(longest_param_name)} : {v}")
        return "\n".join(string)

    def __deepcopy__(self, memo=None):
        return Config(**self.to_dict())
