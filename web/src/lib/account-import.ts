import type { AccountImportPayload } from "./api";


function normalizeAccount(value: unknown): AccountImportPayload | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  const raw = value as Record<string, unknown>;
  const tokenValue = raw.access_token ?? raw.accessToken;
  const accessToken = typeof tokenValue === "string" ? tokenValue.trim() : "";
  if (!accessToken) {
    return null;
  }

  const account: AccountImportPayload = {
    ...raw,
    access_token: accessToken,
    source_type: "codex",
  };
  delete account.accessToken;
  if (account.type === "codex") {
    account.export_type = "codex";
    delete account.type;
  }
  return account;
}

export function parseAccountImportPayload(value: unknown): AccountImportPayload[] {
  if (Array.isArray(value)) {
    return value.map(normalizeAccount).filter((item): item is AccountImportPayload => item !== null);
  }

  const single = normalizeAccount(value);
  if (single) {
    return [single];
  }

  if (!value || typeof value !== "object") {
    return [];
  }
  const raw = value as Record<string, unknown>;
  const nested = raw.accounts ?? raw.items;
  if (!Array.isArray(nested)) {
    return [];
  }
  return nested.map(normalizeAccount).filter((item): item is AccountImportPayload => item !== null);
}
