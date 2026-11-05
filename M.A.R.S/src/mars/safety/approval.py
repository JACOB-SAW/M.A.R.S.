class Approver:
    """Human sign-off gate for changes to the system's purpose.

    mode:
      "auto" - approve (sandbox only; every request is still logged)
      "ask"  - prompt in the terminal
      "deny" - reject everything
    """

    def __init__(self, mode: str = "auto", logger=None):
        if mode not in ("auto", "ask", "deny"):
            raise ValueError(f"unknown approval mode: {mode}")
        self.mode = mode
        self.logger = logger

    def request(self, description: str) -> bool:
        if self.mode == "auto":
            approved = True
        elif self.mode == "deny":
            approved = False
        else:
            answer = input(f"\n[APPROVAL NEEDED] {description}\nApprove? [y/N] ")
            approved = answer.strip().lower() in ("y", "yes")
        if self.logger:
            self.logger.event("approval", description=description,
                              mode=self.mode, approved=approved)
        return approved
