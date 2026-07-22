import { describe, expect, test } from "bun:test";

import { parseAccountImportPayload } from "./account-import";


describe("parseAccountImportPayload", () => {
  test("normalizes a single exported account", () => {
    expect(
      parseAccountImportPayload({
        type: "codex",
        accessToken: " token-1 ",
        refresh_token: "refresh-1",
      }),
    ).toEqual([
      {
        access_token: "token-1",
        export_type: "codex",
        refresh_token: "refresh-1",
        source_type: "codex",
      },
    ]);
  });

  test("accepts an exported account array and skips invalid entries", () => {
    const result = parseAccountImportPayload([
      { access_token: "token-1", email: "one@example.test" },
      null,
      { access_token: "" },
      { accessToken: "token-2" },
    ]);

    expect(result.map((item) => item.access_token)).toEqual(["token-1", "token-2"]);
    expect(result.every((item) => item.source_type === "codex")).toBe(true);
  });

  test("accepts accounts and items wrapper objects", () => {
    expect(parseAccountImportPayload({ accounts: [{ access_token: "token-a" }] })[0]?.access_token).toBe("token-a");
    expect(parseAccountImportPayload({ items: [{ access_token: "token-b" }] })[0]?.access_token).toBe("token-b");
  });

  test("rejects unsupported or empty payloads", () => {
    expect(parseAccountImportPayload({ accounts: { access_token: "token-1" } })).toEqual([]);
    expect(parseAccountImportPayload("token-1")).toEqual([]);
    expect(parseAccountImportPayload([])).toEqual([]);
  });
});
