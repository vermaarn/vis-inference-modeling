export interface CommentReply {
  name: string;
  location: string;
  "date posted": string;
  "comment info": string;
  "reply to"?: string;
}

export interface Comment {
  name: string;
  location: string;
  "date posted": string;
  "comment info": string;
  replies: CommentReply[];
}

export interface UmapRow {
  umap_1: number;
  umap_2: number;
  class: string;
  path: string;
}

export interface ArticleEntry {
  articleUrl: string;
  pngPath: string;
  commentsPath: string;
}
