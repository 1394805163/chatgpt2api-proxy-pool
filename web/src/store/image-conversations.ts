"use client";

import localforage from "localforage";

import type { ImageModel } from "@/lib/api";

export type ImageConversationMode = "generate" | "edit";

export type StoredReferenceImage = {
  name: string;
  type: string;
  dataUrl: string;
};

export type StoredImage = {
  id: string;
  taskId?: string;
  status?: "loading" | "success" | "error";
  taskStatus?: "queued" | "running";
  progress?: string;
  b64_json?: string;
  url?: string;
  revised_prompt?: string;
  error?: string;
  startTime?: number;
  elapsedSecs?: number;
  elapsedUpdatedAt?: number;
  durationMs?: number;
  browserCachedAt?: number;
};

export type ImageTurnStatus = "queued" | "generating" | "success" | "error";

export type ImageTurn = {
  id: string;
  prompt: string;
  model: ImageModel;
  mode: ImageConversationMode;
  referenceImages: StoredReferenceImage[];
  count: number;
  size: string;
  ratio: string;
  tier: string;
  quality: string;
  images: StoredImage[];
  createdAt: string;
  status: ImageTurnStatus;
  error?: string;
  promptDeleted?: boolean;
  resultsDeleted?: boolean;
};

export type ImageConversation = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  turns: ImageTurn[];
};

export type ImageConversationStats = {
  queued: number;
  running: number;
};

export type BrowserCachedImage = {
  cacheKey: string;
  taskId?: string;
  sourceUrl?: string;
  dataUrl: string;
  name: string;
  date: string;
  size: number;
  createdAt: string;
  prompt: string;
};

const imageConversationStorage = localforage.createInstance({
  name: "chatgpt2api",
  storeName: "image_conversations",
});

const IMAGE_CONVERSATIONS_KEY = "items";
let imageConversationWriteQueue: Promise<void> = Promise.resolve();

function normalizeStoredImage(image: StoredImage): StoredImage {
  const normalized = {
    ...image,
    taskId: typeof image.taskId === "string" && image.taskId ? image.taskId : undefined,
    taskStatus: image.taskStatus === "queued" || image.taskStatus === "running" ? image.taskStatus : undefined,
    url: typeof image.url === "string" && image.url ? image.url : undefined,
    revised_prompt: typeof image.revised_prompt === "string" ? image.revised_prompt : undefined,
    startTime: typeof image.startTime === "number" ? image.startTime : undefined,
    elapsedSecs: typeof image.elapsedSecs === "number" ? image.elapsedSecs : undefined,
    elapsedUpdatedAt: typeof image.elapsedUpdatedAt === "number" ? image.elapsedUpdatedAt : undefined,
    durationMs: typeof image.durationMs === "number" ? image.durationMs : undefined,
    browserCachedAt: typeof image.browserCachedAt === "number" ? image.browserCachedAt : undefined,
  };
  if (image.status === "loading" || image.status === "error" || image.status === "success") {
    return normalized;
  }
  return {
    ...normalized,
    status: image.b64_json || image.url ? "success" : "loading",
  };
}

function normalizeReferenceImage(image: StoredReferenceImage): StoredReferenceImage {
  return {
    name: image.name || "reference.png",
    type: image.type || "image/png",
    dataUrl: image.dataUrl,
  };
}

function dataUrlMimeType(dataUrl: string) {
  const match = dataUrl.match(/^data:(.*?);base64,/);
  return match?.[1] || "image/png";
}

function getLegacyReferenceImages(source: Record<string, unknown>): StoredReferenceImage[] {
  if (Array.isArray(source.referenceImages)) {
    return source.referenceImages
      .filter((image): image is StoredReferenceImage => {
        if (!image || typeof image !== "object") {
          return false;
        }
        const candidate = image as StoredReferenceImage;
        return typeof candidate.dataUrl === "string" && candidate.dataUrl.length > 0;
      })
      .map(normalizeReferenceImage);
  }

  if (source.sourceImage && typeof source.sourceImage === "object") {
    const image = source.sourceImage as { dataUrl?: unknown; fileName?: unknown };
    if (typeof image.dataUrl === "string" && image.dataUrl) {
      return [
        {
          name: typeof image.fileName === "string" && image.fileName ? image.fileName : "reference.png",
          type: dataUrlMimeType(image.dataUrl),
          dataUrl: image.dataUrl,
        },
      ];
    }
  }

  return [];
}

function normalizeTurn(turn: ImageTurn & Record<string, unknown>): ImageTurn {
  const normalizedImages = Array.isArray(turn.images) ? turn.images.map(normalizeStoredImage) : [];
  const derivedStatus: ImageTurnStatus =
    normalizedImages.some((image) => image.status === "loading")
      ? "generating"
      : normalizedImages.some((image) => image.status === "error")
        ? "error"
        : "success";

  return {
    id: String(turn.id || `${Date.now()}`),
    prompt: String(turn.prompt || ""),
    model: (turn.model as ImageModel) || "gpt-image-2",
    mode: turn.mode === "edit" ? "edit" : "generate",
    referenceImages: getLegacyReferenceImages(turn),
    count: Math.max(1, Number(turn.count || normalizedImages.length || 1)),
    size: typeof turn.size === "string" ? turn.size : "",
    ratio: typeof turn.ratio === "string" && turn.ratio ? turn.ratio : "1:1",
    tier: typeof turn.tier === "string" && turn.tier ? turn.tier : "1k",
    quality: typeof turn.quality === "string" && turn.quality ? turn.quality : "auto",
    images: normalizedImages,
    createdAt: String(turn.createdAt || new Date().toISOString()),
    status:
      turn.status === "queued" ||
      turn.status === "generating" ||
      turn.status === "success" ||
      turn.status === "error"
        ? turn.status
        : derivedStatus,
    error: typeof turn.error === "string" ? turn.error : undefined,
    promptDeleted: turn.promptDeleted === true,
    resultsDeleted: turn.resultsDeleted === true,
  };
}

function normalizeConversation(conversation: ImageConversation & Record<string, unknown>): ImageConversation {
  const turns = Array.isArray(conversation.turns)
    ? conversation.turns.map((turn) => normalizeTurn(turn as ImageTurn & Record<string, unknown>))
    : [
        normalizeTurn({
          id: String(conversation.id || `${Date.now()}`),
          prompt: String(conversation.prompt || ""),
          model: (conversation.model as ImageModel) || "gpt-image-2",
          mode: conversation.mode === "edit" ? "edit" : "generate",
          referenceImages: getLegacyReferenceImages(conversation),
          count: Number(conversation.count || 1),
          size: typeof conversation.size === "string" ? conversation.size : "",
          ratio: typeof conversation.ratio === "string" && conversation.ratio ? conversation.ratio : "1:1",
          tier: typeof conversation.tier === "string" && conversation.tier ? conversation.tier : "1k",
          quality: typeof conversation.quality === "string" && conversation.quality ? conversation.quality : "auto",
          images: Array.isArray(conversation.images) ? (conversation.images as StoredImage[]) : [],
          createdAt: String(conversation.createdAt || new Date().toISOString()),
          status:
            conversation.status === "generating" || conversation.status === "success" || conversation.status === "error"
              ? conversation.status
              : "success",
          error: typeof conversation.error === "string" ? conversation.error : undefined,
        }),
      ];
  const lastTurn = turns.length > 0 ? turns[turns.length - 1] : null;

  return {
    id: String(conversation.id || `${Date.now()}`),
    title: String(conversation.title || ""),
    createdAt: String(conversation.createdAt || lastTurn?.createdAt || new Date().toISOString()),
    updatedAt: String(conversation.updatedAt || lastTurn?.createdAt || new Date().toISOString()),
    turns,
  };
}

function sortImageConversations(conversations: ImageConversation[]): ImageConversation[] {
  return [...conversations].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

function getTimestamp(value: string) {
  const time = new Date(value).getTime();
  return Number.isFinite(time) ? time : 0;
}

function pickLatestConversation(current: ImageConversation, next: ImageConversation) {
  return getTimestamp(next.updatedAt) >= getTimestamp(current.updatedAt) ? next : current;
}

function queueImageConversationWrite<T>(operation: () => Promise<T>): Promise<T> {
  const result = imageConversationWriteQueue.then(operation);
  imageConversationWriteQueue = result.then(
    () => undefined,
    () => undefined,
  );
  return result;
}

async function readStoredImageConversations(): Promise<ImageConversation[]> {
  const items =
    (await imageConversationStorage.getItem<Array<ImageConversation & Record<string, unknown>>>(
      IMAGE_CONVERSATIONS_KEY,
    )) || [];
  return items.map(normalizeConversation);
}

export async function listImageConversations(): Promise<ImageConversation[]> {
  return sortImageConversations(await readStoredImageConversations());
}

export function imageCacheKeyFromUrl(value: string | undefined): string | null {
  const raw = String(value || "").trim();
  if (!raw || raw.startsWith("data:")) return null;
  try {
    const url = new URL(raw, typeof window === "undefined" ? "https://local.invalid" : window.location.origin);
    const marker = "/images/";
    const markerIndex = url.pathname.indexOf(marker);
    if (markerIndex >= 0) {
      return `image:${decodeURIComponent(url.pathname.slice(markerIndex + marker.length))}`;
    }
    return `url:${url.href}`;
  } catch {
    return `url:${raw}`;
  }
}

function browserCachedImageKey(image: StoredImage): string {
  return imageCacheKeyFromUrl(image.url) || `task:${image.taskId || image.id}`;
}

function base64ByteLength(value: string): number {
  const padding = value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor((value.length * 3) / 4) - padding);
}

function cachedImageName(sourceUrl: string | undefined, fallback: string): string {
  if (!sourceUrl) return `${fallback}.png`;
  try {
    const url = new URL(sourceUrl, typeof window === "undefined" ? "https://local.invalid" : window.location.origin);
    return decodeURIComponent(url.pathname.split("/").filter(Boolean).at(-1) || `${fallback}.png`);
  } catch {
    return `${fallback}.png`;
  }
}

function cachedImageDate(sourceUrl: string | undefined, createdAt: string): string {
  const key = imageCacheKeyFromUrl(sourceUrl);
  const pathDate = key?.match(/^image:(\d{4})\/(\d{2})\/(\d{2})\//);
  if (pathDate) return `${pathDate[1]}-${pathDate[2]}-${pathDate[3]}`;
  return createdAt.slice(0, 10);
}

export async function listBrowserCachedImages(): Promise<BrowserCachedImage[]> {
  const items = await readStoredImageConversations();
  const cached = new Map<string, BrowserCachedImage>();
  for (const conversation of items) {
    for (const turn of conversation.turns) {
      for (const image of turn.images) {
        const b64 = String(image.b64_json || "").trim();
        if (!b64) continue;
        const cacheKey = browserCachedImageKey(image);
        const current = cached.get(cacheKey);
        if (current && current.createdAt >= turn.createdAt) continue;
        cached.set(cacheKey, {
          cacheKey,
          taskId: image.taskId,
          sourceUrl: image.url,
          dataUrl: `data:image/png;base64,${b64}`,
          name: cachedImageName(image.url, image.taskId || image.id || "generated-image"),
          date: cachedImageDate(image.url, turn.createdAt),
          size: base64ByteLength(b64),
          createdAt: turn.createdAt,
          prompt: turn.prompt,
        });
      }
    }
  }
  return [...cached.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function getBrowserCachedImageUrlMap(): Promise<Map<string, string>> {
  return new Map((await listBrowserCachedImages()).map((image) => [image.cacheKey, image.dataUrl]));
}

export async function removeBrowserCachedImages(cacheKeys: string[]): Promise<number> {
  const targets = new Set(cacheKeys);
  if (targets.size === 0) return 0;
  return queueImageConversationWrite(async () => {
    const items = await readStoredImageConversations();
    let removed = 0;
    const nextItems = items.map((conversation) => ({
      ...conversation,
      turns: conversation.turns.map((turn) => ({
        ...turn,
        images: turn.images.map((image) => {
          if (!image.b64_json || !targets.has(browserCachedImageKey(image))) return image;
          removed += 1;
          const nextImage = { ...image };
          delete nextImage.b64_json;
          delete nextImage.browserCachedAt;
          return nextImage;
        }),
      })),
    }));
    if (removed > 0) {
      await imageConversationStorage.setItem(IMAGE_CONVERSATIONS_KEY, sortImageConversations(nextItems));
    }
    return removed;
  });
}

export async function saveImageConversations(conversations: ImageConversation[]): Promise<void> {
  await queueImageConversationWrite(async () => {
    const items = await readStoredImageConversations();
    const conversationMap = new Map(items.map((item) => [item.id, item]));
    for (const conversation of conversations.map(normalizeConversation)) {
      const current = conversationMap.get(conversation.id);
      conversationMap.set(conversation.id, current ? pickLatestConversation(current, conversation) : conversation);
    }
    await imageConversationStorage.setItem(
      IMAGE_CONVERSATIONS_KEY,
      sortImageConversations([...conversationMap.values()]),
    );
  });
}

export async function saveImageConversation(conversation: ImageConversation): Promise<void> {
  await queueImageConversationWrite(async () => {
    const items = await readStoredImageConversations();
    const nextConversation = normalizeConversation(conversation);
    const current = items.find((item) => item.id === nextConversation.id);
    const persistedConversation = current ? pickLatestConversation(current, nextConversation) : nextConversation;
    const nextItems = sortImageConversations([
      persistedConversation,
      ...items.filter((item) => item.id !== persistedConversation.id),
    ]);
    await imageConversationStorage.setItem(IMAGE_CONVERSATIONS_KEY, nextItems);
  });
}

export async function renameImageConversation(id: string, title: string): Promise<void> {
  await queueImageConversationWrite(async () => {
    const items = await readStoredImageConversations();
    const target = items.find((item) => item.id === id);
    if (!target) return;
    const updated = { ...target, title, updatedAt: new Date().toISOString() };
    const nextItems = sortImageConversations([
      updated,
      ...items.filter((item) => item.id !== id),
    ]);
    await imageConversationStorage.setItem(IMAGE_CONVERSATIONS_KEY, nextItems);
  });
}

export async function deleteImageConversation(id: string): Promise<void> {
  await queueImageConversationWrite(async () => {
    const items = await readStoredImageConversations();
    await imageConversationStorage.setItem(
      IMAGE_CONVERSATIONS_KEY,
      items.filter((item) => item.id !== id),
    );
  });
}

export async function clearImageConversations(): Promise<void> {
  await queueImageConversationWrite(async () => {
    await imageConversationStorage.removeItem(IMAGE_CONVERSATIONS_KEY);
  });
}

export function getImageConversationStats(conversation: ImageConversation | null): ImageConversationStats {
  if (!conversation) {
    return { queued: 0, running: 0 };
  }

  return conversation.turns.reduce(
    (acc, turn) => {
      if (turn.resultsDeleted) {
        return acc;
      }
      if (turn.status === "queued") {
        acc.queued += 1;
      } else if (turn.status === "generating") {
        acc.running += 1;
      }
      return acc;
    },
    { queued: 0, running: 0 },
  );
}
