/*-------------------------------------------------------------------------
 *
 * pgEdge Natural Language Agent
 *
 * Copyright (c) 2025 - 2026, pgEdge, Inc.
 * This software is released under The PostgreSQL License
 *
 *-------------------------------------------------------------------------
 */

package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	// GeminiHTTPTimeout is the HTTP client timeout for Gemini API requests
	GeminiHTTPTimeout = 30 * time.Second
	// DefaultGeminiURL is the stable direct endpoint for Google GenAI services
	DefaultGeminiURL  = "https://generativelanguage.googleapis.com"
)

// GeminiProvider implements embedding generation using Google's Gemini API
type GeminiProvider struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
}

// geminiPart matches Google's structural object notation
type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

type geminiEmbeddingRequest struct {
	Content geminiContent `json:"content"`
}

type geminiEmbeddingResponse struct {
	Embedding struct {
		Values []float64 `json:"values"`
	} `json:"embedding"`
}

// Model dimensions for Google embedding models
var geminiModelDimensions = map[string]int{
	"text-embedding-004": 768,
}

// NewGeminiProvider creates a new Gemini embedding provider
// baseURL can be empty to use the default (https://generativelanguage.googleapis.com)
func NewGeminiProvider(apiKey, model, baseURL string) (*GeminiProvider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("Gemini API key cannot be empty")
	}

	// Default to text-embedding-004 if no model specified
	if model == "" {
		model = "text-embedding-004"
	}

	// Validate model is supported
	if _, ok := geminiModelDimensions[model]; !ok {
		return nil, fmt.Errorf("unsupported Gemini model: %s (supported: text-embedding-004)", model)
	}

	// Default base URL if not specified
	if baseURL == "" {
		baseURL = DefaultGeminiURL
	} else {
		// Validate and normalize the base URL
		baseURL = strings.TrimSpace(baseURL)
		baseURL = strings.TrimSuffix(baseURL, "/")

		parsedURL, err := url.Parse(baseURL)
		if err != nil {
			return nil, fmt.Errorf("invalid Gemini base URL: %w", err)
		}
		if parsedURL.Scheme != "https" && parsedURL.Scheme != "http" {
			return nil, fmt.Errorf("Gemini base URL must use http or https scheme, got: %s", parsedURL.Scheme)
		}
		if parsedURL.Host == "" {
			return nil, fmt.Errorf("Gemini base URL must include a host")
		}
	}

	// Mask the API key for logging (show only first/last few characters)
	maskedKey := "(redacted)"
	if len(apiKey) > 8 {
		maskedKey = apiKey[:4] + "..." + apiKey[len(apiKey)-4:]
	}

	LogProviderInit("gemini", model, map[string]string{
		"api_key":  maskedKey,
		"base_url": baseURL,
	})

	return &GeminiProvider{
		apiKey:  apiKey,
		model:   model,
		baseURL: baseURL,
		client: &http.Client{
			Timeout: GeminiHTTPTimeout,
		},
	}, nil
}

// Embed generates an embedding vector for the given text using v1beta Google REST pathways
func (p *GeminiProvider) Embed(ctx context.Context, text string) ([]float64, error) {
	startTime := time.Now()
	textLen := len(text)

	if text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Format endpoint structure: /v1beta/models/{model}:embedContent
	targetEndpoint := fmt.Sprintf("%s/v1beta/models/%s:embedContent", p.baseURL, p.model)
	
	u, err := url.Parse(targetEndpoint)
	if err != nil {
		duration := time.Since(startTime)
		LogAPICall("gemini", p.model, textLen, duration, 0, err)
		return nil, fmt.Errorf("failed to parse target gemini url path: %w", err)
	}

	// Append API Key explicitly via URI Query Parameter as mandated by Google Dev guidelines
	q := u.Query()
	q.Set("key", p.apiKey)
	u.RawQuery = q.Encode()

	LogAPICallDetails("gemini", p.model, targetEndpoint, textLen)
	LogRequestTrace("gemini", p.model, text)

	// Assemble matching nested layout payload blocks
	reqBody := geminiEmbeddingRequest{
		Content: geminiContent{
			Parts: []geminiPart{
				{Text: text},
			},
		},
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		duration := time.Since(startTime)
		LogAPICall("gemini", p.model, textLen, duration, 0, err)
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", u.String(), bytes.NewReader(reqBytes))
	if err != nil {
		duration := time.Since(startTime)
		LogAPICall("gemini", p.model, textLen, duration, 0, err)
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		LogConnectionError("gemini", u.String(), err)
		duration := time.Since(startTime)
		LogAPICall("gemini", p.model, textLen, duration, 0, err)
		return nil, fmt.Errorf("failed to make API request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			duration := time.Since(startTime)
			err := fmt.Errorf("API request failed with status %d (error reading response body: %w)", resp.StatusCode, readErr)
			LogAPICall("gemini", p.model, textLen, duration, 0, err)
			return nil, err
		}

		// Check if this is a rate limit error (429)
		if resp.StatusCode == 429 {
			LogRateLimitError("gemini", p.model, resp.StatusCode, string(body))
		}

		duration := time.Since(startTime)
		err := fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
		LogAPICall("gemini", p.model, textLen, duration, 0, err)
		return nil, err
	}

	var embResp geminiEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		duration := time.Since(startTime)
		LogAPICall("gemini", p.model, textLen, duration, 0, err)
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(embResp.Embedding.Values) == 0 {
		duration := time.Since(startTime)
		err := fmt.Errorf("received empty embedding from API")
		LogAPICall("gemini", p.model, textLen, duration, 0, err)
		return nil, err
	}

	duration := time.Since(startTime)
	dimensions := len(embResp.Embedding.Values)
	LogResponseTrace("gemini", p.model, resp.StatusCode, dimensions)
	LogAPICall("gemini", p.model, textLen, duration, dimensions, nil)

	return embResp.Embedding.Values, nil
}

// Dimensions returns the number of dimensions for this model mapped safely from the configuration array
func (p *GeminiProvider) Dimensions() int {
	return geminiModelDimensions[p.model]
}

// ModelName returns the model name
func (p *GeminiProvider) ModelName() string {
	return p.model
}

// ProviderName returns "gemini"
func (p *GeminiProvider) ProviderName() string {
	return "gemini"
}