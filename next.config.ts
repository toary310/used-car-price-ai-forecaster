import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  async headers() {
    return [
      {
        source: '/:path*', // すべてのページに適用
        headers: [
          {
            key: 'Content-Security-Policy',
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline'", // Next.jsの機能のために必要
              "style-src 'self' 'unsafe-inline'",  // スタイリングのために必要
              "img-src 'self' data: blob:",         // 画像用
              "font-src 'self'",                    // フォント用
              "connect-src 'self' http://localhost:8000", // APIリクエスト用
              "frame-ancestors 'none'",             // クリックジャッキング対策
              "form-action 'self'",                 // フォーム送信先制限
              "base-uri 'self'",                    // base hrefタグの制限
              "object-src 'none'",                  // 埋め込みオブジェクト制限
            ].join('; ')
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'                      // MIMEタイプスニッフィング防止
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY'                         // クリックジャッキング対策
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block'                // ブラウザのXSSフィルター有効化
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin' // リファラー制限
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()' // 権限制限
          }
        ]
      }
    ];
  }
};

export default nextConfig;
