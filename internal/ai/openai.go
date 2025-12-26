package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const openaiAPIURL = "https://api.openai.com/v1/chat/completions"

// OpenAIProvider はOpenAI APIを使用したプロバイダー
type OpenAIProvider struct {
	apiKey string
	model  string
}

// NewOpenAIProvider は新しいOpenAIProviderを作成する
func NewOpenAIProvider(apiKey string) (*OpenAIProvider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	return &OpenAIProvider{
		apiKey: apiKey,
		model:  "gpt-4o",
	}, nil
}

// Name はプロバイダー名を返す
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// openaiRequest はOpenAI APIへのリクエスト
type openaiRequest struct {
	Model       string          `json:"model"`
	Messages    []openaiMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
}

type openaiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// openaiResponse はOpenAI APIからのレスポンス
type openaiResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

// Verify はSPECとコードの一致度を検証する
func (p *OpenAIProvider) Verify(ctx context.Context, specContent string, codeContents map[string]string) (*VerificationResult, error) {
	return p.VerifyWithOptions(ctx, specContent, codeContents, nil)
}

// VerifyWithOptions は検証観点を指定してSPECとコードの一致度を検証する
func (p *OpenAIProvider) VerifyWithOptions(ctx context.Context, specContent string, codeContents map[string]string, opts *VerifyOptions) (*VerificationResult, error) {
	var prompt string
	if opts != nil && len(opts.VerificationFocus) > 0 {
		prompt = buildVerificationPromptWithFocus(specContent, codeContents, opts.VerificationFocus)
	} else {
		prompt = buildVerificationPrompt(specContent, codeContents)
	}

	text, err := p.callAPI(ctx, prompt, 2000)
	if err != nil {
		return nil, err
	}

	return parseVerificationResult(text)
}

// ExtractEndpoints はコードからAPIエンドポイント/ページルートを抽出する
func (p *OpenAIProvider) ExtractEndpoints(ctx context.Context, opts *ExtractOptions, codeContent string) ([]EndpointResult, error) {
	var prompt string
	if opts.IsUICategory() {
		prompt = buildUIRouteExtractionPrompt(opts.GetSourceType(), codeContent)
	} else {
		prompt = buildEndpointExtractionPrompt(opts.GetSourceType(), codeContent)
	}

	text, err := p.callAPI(ctx, prompt, 4000)
	if err != nil {
		return nil, err
	}

	return parseEndpointResult(text)
}

// callAPI はOpenAI APIを呼び出す共通関数
func (p *OpenAIProvider) callAPI(ctx context.Context, prompt string, maxTokens int) (string, error) {
	req := openaiRequest{
		Model:       p.model,
		MaxTokens:   maxTokens,
		Temperature: 0.1,
		Messages: []openaiMessage{
			{Role: "user", Content: prompt},
		},
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", openaiAPIURL, bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var openaiResp openaiResponse
	if err := json.Unmarshal(body, &openaiResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	if openaiResp.Error != nil {
		return "", fmt.Errorf("API error: %s", openaiResp.Error.Message)
	}

	if len(openaiResp.Choices) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	return openaiResp.Choices[0].Message.Content, nil
}
