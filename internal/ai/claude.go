package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
)

const claudeAPIURL = "https://api.anthropic.com/v1/messages"

// ClaudeProvider はClaude APIを使用したプロバイダー
type ClaudeProvider struct {
	apiKey string
	model  string
}

// NewClaudeProvider は新しいClaudeProviderを作成する
func NewClaudeProvider(apiKey string) (*ClaudeProvider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	return &ClaudeProvider{
		apiKey: apiKey,
		model:  "claude-sonnet-4-20250514",
	}, nil
}

// Name はプロバイダー名を返す
func (p *ClaudeProvider) Name() string {
	return "claude"
}

// claudeRequest はClaude APIへのリクエスト
type claudeRequest struct {
	Model     string          `json:"model"`
	MaxTokens int             `json:"max_tokens"`
	Messages  []claudeMessage `json:"messages"`
}

type claudeMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// claudeResponse はClaude APIからのレスポンス
type claudeResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Verify はSPECとコードの一致度を検証する
func (p *ClaudeProvider) Verify(ctx context.Context, specContent string, codeContents map[string]string) (*VerificationResult, error) {
	prompt := buildVerificationPrompt(specContent, codeContents)

	req := claudeRequest{
		Model:     p.model,
		MaxTokens: 2000,
		Messages: []claudeMessage{
			{Role: "user", Content: prompt},
		},
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", claudeAPIURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var claudeResp claudeResponse
	if err := json.Unmarshal(body, &claudeResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if claudeResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", claudeResp.Error.Message)
	}

	if len(claudeResp.Content) == 0 {
		return nil, fmt.Errorf("empty response from API")
	}

	return parseVerificationResult(claudeResp.Content[0].Text)
}

// buildVerificationPrompt は検証用のプロンプトを構築する
func buildVerificationPrompt(specContent string, codeContents map[string]string) string {
	var codeSection strings.Builder
	for filePath, content := range codeContents {
		codeSection.WriteString(fmt.Sprintf("\n### %s\n```\n%s\n```\n", filePath, content))
	}

	return fmt.Sprintf(`あなたはコードレビューの専門家です。以下のSPEC（仕様書）と実際のコードを比較して、一致度を評価してください。

## SPEC（仕様書）
%s

## 実際のコード
%s

## 評価基準
以下の観点で評価してください：
1. 画面構成: SPECに記載された要素がコードに存在するか
2. 状態管理: SPECに記載された状態やフックが使用されているか
3. 処理フロー: SPECに記載された処理フローがコードで実装されているか
4. バリデーション: SPECに記載されたバリデーションルールが実装されているか
5. エラーハンドリング: SPECに記載されたエラーケースが処理されているか

## 出力形式
以下のJSON形式で出力してください：
%sjson
{
  "matchPercentage": <0-100の数値>,
  "matchedItems": ["一致している項目1", "一致している項目2", ...],
  "unmatchedItems": ["一致していない項目1", "一致していない項目2", ...],
  "notes": "補足コメント（未実装の機能や改善点など）"
}
%s

JSONのみを出力してください。`, specContent, codeSection.String(), "```", "```")
}

// parseVerificationResult はClaude APIのレスポンスから検証結果を抽出する
func parseVerificationResult(text string) (*VerificationResult, error) {
	// JSONブロックを抽出
	jsonRegex := regexp.MustCompile("```json\\s*([\\s\\S]*?)\\s*```")
	matches := jsonRegex.FindStringSubmatch(text)

	var jsonStr string
	if len(matches) >= 2 {
		jsonStr = matches[1]
	} else {
		// JSONブロックがない場合は直接パースを試みる
		jsonStr = text
	}

	var result VerificationResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("failed to parse verification result: %w", err)
	}

	return &result, nil
}
