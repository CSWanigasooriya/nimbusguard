# OpenAI API Key Management

## Overview

NimbusGuard includes convenient Makefile utilities to manage OpenAI API keys for the LangGraph operator. This prevents the common issue where the API key reverts to a placeholder after deployments.

## Quick Setup

After deploying NimbusGuard, set up your OpenAI API key:

```bash
make setup-openai
```

This will:
- Prompt for your OpenAI API key (must start with `sk-`)
- Update the Kubernetes secret
- Restart the LangGraph operator to pick up the new key

## Troubleshooting

If the LangGraph operator is in CrashLoopBackOff:

```bash
make fix-openai
```

This will:
- Check the current API key status
- Restart the operator if the key is valid
- Provide guidance if the key needs to be set

## Integration with Deployment

The deployment process (`make k8s-deploy`) automatically checks the OpenAI API key status and warns if it's a placeholder:

```
🔑 Checking OpenAI API key...
   ⚠️  OpenAI API key is placeholder - operator will fail
   💡 Run 'make setup-openai' after deployment to fix
```

## Manual Management

You can also manage the API key manually:

```bash
# Check current key (first 8 characters shown)
kubectl get secret openai-secret -n nimbusguard -o jsonpath='{.data.api-key}' | base64 -d

# Update key manually
kubectl patch secret openai-secret -n nimbusguard -p '{"data":{"api-key":"'$(echo -n "sk-your-key-here" | base64)'"}}
'

# Restart operator
kubectl rollout restart deployment/langgraph-operator -n nimbusguard
```

## Security Best Practices

- Never commit API keys to version control
- Use separate API keys for development and production
- Rotate API keys regularly
- Monitor API usage through OpenAI dashboard

## Common Issues

**Issue**: LangGraph operator keeps crashing with "Invalid API key"
**Solution**: Run `make setup-openai` to set a proper development key

**Issue**: API key reverts to placeholder after redeployment
**Solution**: The Makefile utilities prevent this by updating the secret directly, bypassing the manifest placeholders

**Issue**: "Invalid API key format" error
**Solution**: Ensure your API key starts with `sk-` and is from your OpenAI account

## Available Commands

- `make setup-openai` - Interactive setup of OpenAI API key
- `make fix-openai` - Diagnose and fix API key issues
- `make k8s-deploy` - Includes automatic API key validation 